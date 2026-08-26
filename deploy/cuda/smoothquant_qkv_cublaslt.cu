#include <cublasLt.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace
{

constexpr int32_t kM = 177;
constexpr int32_t kK = 960;
constexpr int32_t kNQ = 960;
constexpr int32_t kNKV = 320;
constexpr size_t kWorkspaceBytes = 16 * 1024 * 1024;

void checkCuda(cudaError_t err, char const* what)
{
    if (err != cudaSuccess)
    {
        std::fprintf(stderr, "%s failed: %s\n", what, cudaGetErrorString(err));
        std::exit(1);
    }
}

void checkCublas(cublasStatus_t status, char const* what)
{
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        std::fprintf(stderr, "%s failed: cublas status %d\n", what, static_cast<int>(status));
        std::exit(1);
    }
}

std::vector<char> readBytes(std::filesystem::path const& path, size_t expected)
{
    std::ifstream is(path, std::ios::binary);
    if (!is)
    {
        std::fprintf(stderr, "failed to open %s\n", path.c_str());
        std::exit(1);
    }
    std::vector<char> data(expected);
    is.read(data.data(), static_cast<std::streamsize>(data.size()));
    if (static_cast<size_t>(is.gcount()) != expected)
    {
        std::fprintf(stderr, "unexpected size for %s: got %lld expected %zu\n", path.c_str(), static_cast<long long>(is.gcount()), expected);
        std::exit(1);
    }
    return data;
}

std::vector<float> readFloatFile(std::filesystem::path const& path, size_t count)
{
    auto bytes = readBytes(path, count * sizeof(float));
    std::vector<float> out(count);
    std::memcpy(out.data(), bytes.data(), bytes.size());
    return out;
}

std::vector<int8_t> readInt8File(std::filesystem::path const& path, size_t count)
{
    auto bytes = readBytes(path, count * sizeof(int8_t));
    std::vector<int8_t> out(count);
    std::memcpy(out.data(), bytes.data(), bytes.size());
    return out;
}

struct Metrics
{
    double cosine{};
    double relativeL2{};
    double maxAbs{};
};

Metrics computeMetrics(std::vector<float> const& ref, std::vector<float> const& got)
{
    double dot = 0.0;
    double ref2 = 0.0;
    double got2 = 0.0;
    double diff2 = 0.0;
    double maxAbs = 0.0;
    for (size_t i = 0; i < ref.size(); ++i)
    {
        double a = static_cast<double>(ref[i]);
        double b = static_cast<double>(got[i]);
        double d = b - a;
        dot += a * b;
        ref2 += a * a;
        got2 += b * b;
        diff2 += d * d;
        maxAbs = std::max(maxAbs, std::abs(d));
    }
    double denom = std::sqrt(std::max(ref2 * got2, 1.0e-24));
    return {dot / denom, std::sqrt(diff2) / std::sqrt(std::max(ref2, 1.0e-24)), maxAbs};
}

__device__ int8_t quantizeInt8(float value)
{
    int32_t q = static_cast<int32_t>(nearbyintf(value));
    q = max(-128, min(127, q));
    return static_cast<int8_t>(q);
}

__global__ void quantizeKernel(float const* x, float const* smoothScale, int8_t* qx, float activationScale)
{
    int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= kM * kK)
    {
        return;
    }
    int32_t k = idx % kK;
    qx[idx] = quantizeInt8((x[idx] / smoothScale[k]) / activationScale);
}

__global__ void dequantKernel(int32_t const* acc, float const* ws, float const* bias, float* out, float as, int32_t n)
{
    int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= kM * n)
    {
        return;
    }
    int32_t col = idx % n;
    out[idx] = static_cast<float>(acc[idx]) * as * ws[col] + bias[col];
}

__global__ void dequantQkvFusedKernel(int32_t const* qAcc, int32_t const* kAcc, int32_t const* vAcc,
    float const* qWs, float const* kWs, float const* vWs, float const* qBias, float const* kBias, float const* vBias,
    float* qOut, float* kOut, float* vOut, float qAs, float kAs, float vAs)
{
    int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t qTotal = kM * kNQ;
    int32_t kvTotal = kM * kNKV;
    int32_t total = qTotal + 2 * kvTotal;
    if (idx >= total)
    {
        return;
    }
    if (idx < qTotal)
    {
        int32_t col = idx % kNQ;
        qOut[idx] = static_cast<float>(qAcc[idx]) * qAs * qWs[col] + qBias[col];
    }
    else if (idx < qTotal + kvTotal)
    {
        int32_t local = idx - qTotal;
        int32_t col = local % kNKV;
        kOut[local] = static_cast<float>(kAcc[local]) * kAs * kWs[col] + kBias[col];
    }
    else
    {
        int32_t local = idx - qTotal - kvTotal;
        int32_t col = local % kNKV;
        vOut[local] = static_cast<float>(vAcc[local]) * vAs * vWs[col] + vBias[col];
    }
}

struct LtPlan
{
    cublasLtMatmulDesc_t op{};
    cublasLtMatrixLayout_t a{};
    cublasLtMatrixLayout_t b{};
    cublasLtMatrixLayout_t c{};
    cublasLtMatrixLayout_t d{};
    cublasLtMatmulPreference_t pref{};
    cublasLtMatmulHeuristicResult_t heuristic{};
    int32_t n{};
};

struct LtContext
{
    cublasLtHandle_t handle{};
    LtPlan q{};
    LtPlan kv{};
};

void destroyPlan(LtPlan& p)
{
    if (p.pref) cublasLtMatmulPreferenceDestroy(p.pref);
    if (p.d) cublasLtMatrixLayoutDestroy(p.d);
    if (p.c) cublasLtMatrixLayoutDestroy(p.c);
    if (p.b) cublasLtMatrixLayoutDestroy(p.b);
    if (p.a) cublasLtMatrixLayoutDestroy(p.a);
    if (p.op) cublasLtMatmulDescDestroy(p.op);
}

LtPlan makePlan(cublasLtHandle_t handle, int32_t n)
{
    LtPlan p;
    p.n = n;
    checkCublas(cublasLtMatmulDescCreate(&p.op, CUBLAS_COMPUTE_32I, CUDA_R_32I), "desc");
    cublasOperation_t transA = CUBLAS_OP_N;
    cublasOperation_t transB = CUBLAS_OP_T;
    checkCublas(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSA, &transA, sizeof(transA)), "transA");
    checkCublas(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSB, &transB, sizeof(transB)), "transB");
    checkCublas(cublasLtMatrixLayoutCreate(&p.a, CUDA_R_8I, kM, kK, kK), "a");
    checkCublas(cublasLtMatrixLayoutCreate(&p.b, CUDA_R_8I, n, kK, kK), "b");
    checkCublas(cublasLtMatrixLayoutCreate(&p.c, CUDA_R_32I, kM, n, n), "c");
    checkCublas(cublasLtMatrixLayoutCreate(&p.d, CUDA_R_32I, kM, n, n), "d");
    cublasLtOrder_t row = CUBLASLT_ORDER_ROW;
    checkCublas(cublasLtMatrixLayoutSetAttribute(p.a, CUBLASLT_MATRIX_LAYOUT_ORDER, &row, sizeof(row)), "a row");
    checkCublas(cublasLtMatrixLayoutSetAttribute(p.b, CUBLASLT_MATRIX_LAYOUT_ORDER, &row, sizeof(row)), "b row");
    checkCublas(cublasLtMatrixLayoutSetAttribute(p.c, CUBLASLT_MATRIX_LAYOUT_ORDER, &row, sizeof(row)), "c row");
    checkCublas(cublasLtMatrixLayoutSetAttribute(p.d, CUBLASLT_MATRIX_LAYOUT_ORDER, &row, sizeof(row)), "d row");
    checkCublas(cublasLtMatmulPreferenceCreate(&p.pref), "pref");
    checkCublas(cublasLtMatmulPreferenceSetAttribute(p.pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &kWorkspaceBytes, sizeof(kWorkspaceBytes)), "workspace");
    int32_t returned = 0;
    checkCublas(cublasLtMatmulAlgoGetHeuristic(handle, p.op, p.a, p.b, p.c, p.d, p.pref, 1, &p.heuristic, &returned), "heuristic");
    if (returned <= 0 || p.heuristic.state != CUBLAS_STATUS_SUCCESS)
    {
        std::fprintf(stderr, "no cublasLt algorithm for n=%d\n", n);
        std::exit(1);
    }
    return p;
}

float* gX{};
float *gQSmooth{}, *gKSmooth{}, *gVSmooth{};
int8_t *gQx{}, *gKx{}, *gVx{};
int8_t *gQw{}, *gKw{}, *gVw{};
int32_t *gQAcc{}, *gKAcc{}, *gVAcc{};
float *gQWs{}, *gKWs{}, *gVWs{}, *gQBias{}, *gKBias{}, *gVBias{}, *gQOut{}, *gKOut{}, *gVOut{};
float gQAs{}, gKAs{}, gVAs{};
void* gWorkspace{};
LtContext gLt{};

void matmul(cublasLtHandle_t handle, LtPlan& p, int8_t* a, int8_t* b, int32_t* d, cudaStream_t stream)
{
    int32_t alpha = 1;
    int32_t beta = 0;
    checkCublas(cublasLtMatmul(handle, p.op, &alpha, a, p.a, b, p.b, &beta, d, p.c, d, p.d, &p.heuristic.algo,
                    gWorkspace, kWorkspaceBytes, stream),
        "cublasLtMatmul");
}

void launchBaseline(cudaStream_t stream)
{
    quantizeKernel<<<(kM * kK + 255) / 256, 256, 0, stream>>>(gX, gQSmooth, gQx, gQAs);
    quantizeKernel<<<(kM * kK + 255) / 256, 256, 0, stream>>>(gX, gKSmooth, gKx, gKAs);
    quantizeKernel<<<(kM * kK + 255) / 256, 256, 0, stream>>>(gX, gVSmooth, gVx, gVAs);
    matmul(gLt.handle, gLt.q, gQx, gQw, gQAcc, stream);
    matmul(gLt.handle, gLt.kv, gKx, gKw, gKAcc, stream);
    matmul(gLt.handle, gLt.kv, gVx, gVw, gVAcc, stream);
    dequantKernel<<<(kM * kNQ + 255) / 256, 256, 0, stream>>>(gQAcc, gQWs, gQBias, gQOut, gQAs, kNQ);
    dequantKernel<<<(kM * kNKV + 255) / 256, 256, 0, stream>>>(gKAcc, gKWs, gKBias, gKOut, gKAs, kNKV);
    dequantKernel<<<(kM * kNKV + 255) / 256, 256, 0, stream>>>(gVAcc, gVWs, gVBias, gVOut, gVAs, kNKV);
}

void launchFusedDequant(cudaStream_t stream)
{
    quantizeKernel<<<(kM * kK + 255) / 256, 256, 0, stream>>>(gX, gQSmooth, gQx, gQAs);
    quantizeKernel<<<(kM * kK + 255) / 256, 256, 0, stream>>>(gX, gKSmooth, gKx, gKAs);
    quantizeKernel<<<(kM * kK + 255) / 256, 256, 0, stream>>>(gX, gVSmooth, gVx, gVAs);
    matmul(gLt.handle, gLt.q, gQx, gQw, gQAcc, stream);
    matmul(gLt.handle, gLt.kv, gKx, gKw, gKAcc, stream);
    matmul(gLt.handle, gLt.kv, gVx, gVw, gVAcc, stream);
    int32_t total = kM * (kNQ + 2 * kNKV);
    dequantQkvFusedKernel<<<(total + 255) / 256, 256, 0, stream>>>(gQAcc, gKAcc, gVAcc, gQWs, gKWs, gVWs,
        gQBias, gKBias, gVBias, gQOut, gKOut, gVOut, gQAs, gKAs, gVAs);
}

float timeCuda(cudaStream_t stream, int32_t warmup, int32_t iters, void (*fn)(cudaStream_t))
{
    for (int32_t i = 0; i < warmup; ++i) fn(stream);
    checkCuda(cudaStreamSynchronize(stream), "warmup sync");
    cudaEvent_t start{}, stop{};
    checkCuda(cudaEventCreate(&start), "event start");
    checkCuda(cudaEventCreate(&stop), "event stop");
    checkCuda(cudaEventRecord(start, stream), "record start");
    for (int32_t i = 0; i < iters; ++i) fn(stream);
    checkCuda(cudaEventRecord(stop, stream), "record stop");
    checkCuda(cudaEventSynchronize(stop), "sync stop");
    float ms = 0.0F;
    checkCuda(cudaEventElapsedTime(&ms, start, stop), "elapsed");
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms / static_cast<float>(std::max(1, iters));
}

void mallocCopyFloat(float** dst, std::vector<float> const& h, char const* name)
{
    checkCuda(cudaMalloc(dst, h.size() * sizeof(float)), name);
    checkCuda(cudaMemcpy(*dst, h.data(), h.size() * sizeof(float), cudaMemcpyHostToDevice), name);
}

void mallocCopyInt8(int8_t** dst, std::vector<int8_t> const& h, char const* name)
{
    checkCuda(cudaMalloc(dst, h.size() * sizeof(int8_t)), name);
    checkCuda(cudaMemcpy(*dst, h.data(), h.size() * sizeof(int8_t), cudaMemcpyHostToDevice), name);
}

} // namespace

int main(int argc, char** argv)
{
    std::filesystem::path dataDir = argc > 1 ? argv[1] : "runs/deploy/6/H-qkv-smoothquant-cublaslt/tensors";
    std::filesystem::path outDir = argc > 2 ? argv[2] : "runs/deploy/6/H-qkv-smoothquant-cublaslt";
    gQAs = argc > 3 ? std::atof(argv[3]) : 1.0F;
    gKAs = argc > 4 ? std::atof(argv[4]) : 1.0F;
    gVAs = argc > 5 ? std::atof(argv[5]) : 1.0F;
    int32_t warmup = argc > 6 ? std::atoi(argv[6]) : 20;
    int32_t iters = argc > 7 ? std::atoi(argv[7]) : 200;
    std::filesystem::create_directories(outDir);

    auto hX = readFloatFile(dataDir / "x.fp32.bin", static_cast<size_t>(kM) * kK);
    auto hQS = readFloatFile(dataDir / "q_smooth_scale.fp32.bin", kK);
    auto hKS = readFloatFile(dataDir / "k_smooth_scale.fp32.bin", kK);
    auto hVS = readFloatFile(dataDir / "v_smooth_scale.fp32.bin", kK);
    auto hQw = readInt8File(dataDir / "q_qweight.int8.bin", static_cast<size_t>(kNQ) * kK);
    auto hKw = readInt8File(dataDir / "k_qweight.int8.bin", static_cast<size_t>(kNKV) * kK);
    auto hVw = readInt8File(dataDir / "v_qweight.int8.bin", static_cast<size_t>(kNKV) * kK);
    auto hQWs = readFloatFile(dataDir / "q_weight_scale.fp32.bin", kNQ);
    auto hKWs = readFloatFile(dataDir / "k_weight_scale.fp32.bin", kNKV);
    auto hVWs = readFloatFile(dataDir / "v_weight_scale.fp32.bin", kNKV);
    auto hQBias = readFloatFile(dataDir / "q_bias.fp32.bin", kNQ);
    auto hKBias = readFloatFile(dataDir / "k_bias.fp32.bin", kNKV);
    auto hVBias = readFloatFile(dataDir / "v_bias.fp32.bin", kNKV);
    auto hQRef = readFloatFile(dataDir / "q_reference.fp32.bin", static_cast<size_t>(kM) * kNQ);
    auto hKRef = readFloatFile(dataDir / "k_reference.fp32.bin", static_cast<size_t>(kM) * kNKV);
    auto hVRef = readFloatFile(dataDir / "v_reference.fp32.bin", static_cast<size_t>(kM) * kNKV);

    mallocCopyFloat(&gX, hX, "x");
    mallocCopyFloat(&gQSmooth, hQS, "q smooth");
    mallocCopyFloat(&gKSmooth, hKS, "k smooth");
    mallocCopyFloat(&gVSmooth, hVS, "v smooth");
    mallocCopyInt8(&gQw, hQw, "qw");
    mallocCopyInt8(&gKw, hKw, "kw");
    mallocCopyInt8(&gVw, hVw, "vw");
    mallocCopyFloat(&gQWs, hQWs, "q ws");
    mallocCopyFloat(&gKWs, hKWs, "k ws");
    mallocCopyFloat(&gVWs, hVWs, "v ws");
    mallocCopyFloat(&gQBias, hQBias, "q bias");
    mallocCopyFloat(&gKBias, hKBias, "k bias");
    mallocCopyFloat(&gVBias, hVBias, "v bias");
    checkCuda(cudaMalloc(&gQx, static_cast<size_t>(kM) * kK), "q x");
    checkCuda(cudaMalloc(&gKx, static_cast<size_t>(kM) * kK), "k x");
    checkCuda(cudaMalloc(&gVx, static_cast<size_t>(kM) * kK), "v x");
    checkCuda(cudaMalloc(&gQAcc, static_cast<size_t>(kM) * kNQ * sizeof(int32_t)), "q acc");
    checkCuda(cudaMalloc(&gKAcc, static_cast<size_t>(kM) * kNKV * sizeof(int32_t)), "k acc");
    checkCuda(cudaMalloc(&gVAcc, static_cast<size_t>(kM) * kNKV * sizeof(int32_t)), "v acc");
    checkCuda(cudaMalloc(&gQOut, static_cast<size_t>(kM) * kNQ * sizeof(float)), "q out");
    checkCuda(cudaMalloc(&gKOut, static_cast<size_t>(kM) * kNKV * sizeof(float)), "k out");
    checkCuda(cudaMalloc(&gVOut, static_cast<size_t>(kM) * kNKV * sizeof(float)), "v out");
    checkCuda(cudaMalloc(&gWorkspace, kWorkspaceBytes), "workspace");

    checkCublas(cublasLtCreate(&gLt.handle), "lt create");
    gLt.q = makePlan(gLt.handle, kNQ);
    gLt.kv = makePlan(gLt.handle, kNKV);
    cudaStream_t stream{};
    checkCuda(cudaStreamCreate(&stream), "stream");
    float baselineMs = timeCuda(stream, warmup, iters, launchBaseline);
    float fusedMs = timeCuda(stream, warmup, iters, launchFusedDequant);

    std::vector<float> hQOut(static_cast<size_t>(kM) * kNQ);
    std::vector<float> hKOut(static_cast<size_t>(kM) * kNKV);
    std::vector<float> hVOut(static_cast<size_t>(kM) * kNKV);
    checkCuda(cudaMemcpy(hQOut.data(), gQOut, hQOut.size() * sizeof(float), cudaMemcpyDeviceToHost), "q out copy");
    checkCuda(cudaMemcpy(hKOut.data(), gKOut, hKOut.size() * sizeof(float), cudaMemcpyDeviceToHost), "k out copy");
    checkCuda(cudaMemcpy(hVOut.data(), gVOut, hVOut.size() * sizeof(float), cudaMemcpyDeviceToHost), "v out copy");
    Metrics qm = computeMetrics(hQRef, hQOut);
    Metrics km = computeMetrics(hKRef, hKOut);
    Metrics vm = computeMetrics(hVRef, hVOut);

    std::ofstream report(outDir / "smoothquant_qkv_cublaslt_report.json");
    report << "{\n"
           << "  \"target\": \"layer0.self_attn.qkv_proj\",\n"
           << "  \"m\": " << kM << ", \"k\": " << kK << ", \"n_q\": " << kNQ << ", \"n_kv\": " << kNKV << ",\n"
           << "  \"warmup\": " << warmup << ", \"iters\": " << iters << ",\n"
           << "  \"latency_ms\": {\n"
           << "    \"baseline_three_quant_three_gemm_three_dequant\": " << baselineMs << ",\n"
           << "    \"fused_dequant_three_quant_three_gemm_one_dequant\": " << fusedMs << "\n"
           << "  },\n"
           << "  \"speedup\": {\"fused_dequant_vs_baseline\": " << (baselineMs / std::max(fusedMs, 1.0e-12F)) << "},\n"
           << "  \"metrics\": {\n"
           << "    \"q\": {\"cosine\": " << qm.cosine << ", \"relative_l2\": " << qm.relativeL2 << ", \"max_abs\": " << qm.maxAbs << "},\n"
           << "    \"k\": {\"cosine\": " << km.cosine << ", \"relative_l2\": " << km.relativeL2 << ", \"max_abs\": " << km.maxAbs << "},\n"
           << "    \"v\": {\"cosine\": " << vm.cosine << ", \"relative_l2\": " << vm.relativeL2 << ", \"max_abs\": " << vm.maxAbs << "}\n"
           << "  }\n"
           << "}\n";
    report.close();
    std::printf("baseline qkv: %.6f ms\n", baselineMs);
    std::printf("fused dequant qkv: %.6f ms\n", fusedMs);
    std::printf("speedup: %.6fx\n", baselineMs / std::max(fusedMs, 1.0e-12F));
    std::printf("q rel_l2 %.9g k rel_l2 %.9g v rel_l2 %.9g\n", qm.relativeL2, km.relativeL2, vm.relativeL2);

    cudaStreamDestroy(stream);
    destroyPlan(gLt.q);
    destroyPlan(gLt.kv);
    cublasLtDestroy(gLt.handle);
    return 0;
}
