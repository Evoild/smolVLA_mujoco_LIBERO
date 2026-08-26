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
constexpr int32_t kN = 960;
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
        std::fprintf(stderr, "unexpected size for %s: got %lld expected %zu\n", path.c_str(),
            static_cast<long long>(is.gcount()), expected);
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

void writeFloatFile(std::filesystem::path const& path, std::vector<float> const& values)
{
    std::ofstream os(path, std::ios::binary);
    os.write(reinterpret_cast<char const*>(values.data()), static_cast<std::streamsize>(values.size() * sizeof(float)));
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

__global__ void smoothQuantActivationKernel(
    float const* x, float const* smoothScale, int8_t* qx, float activationScale, int32_t total)
{
    int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total)
    {
        return;
    }
    int32_t k = idx % kK;
    float xSmooth = x[idx] / smoothScale[k];
    qx[idx] = quantizeInt8(xSmooth / activationScale);
}

__global__ void smoothQuantActivationVec4Kernel(float const* x, float const* smoothScale, int8_t* qx, float activationScale)
{
    int32_t packIdx = blockIdx.x * blockDim.x + threadIdx.x;
    constexpr int32_t packs = (kM * kK) / 4;
    if (packIdx >= packs)
    {
        return;
    }
    int32_t base = packIdx * 4;
    int32_t k = base % kK;
    float4 xv = reinterpret_cast<float4 const*>(x)[packIdx];
    float4 sv = *reinterpret_cast<float4 const*>(smoothScale + k);
    char4 out{};
    out.x = quantizeInt8((xv.x / sv.x) / activationScale);
    out.y = quantizeInt8((xv.y / sv.y) / activationScale);
    out.z = quantizeInt8((xv.z / sv.z) / activationScale);
    out.w = quantizeInt8((xv.w / sv.w) / activationScale);
    reinterpret_cast<char4*>(qx)[packIdx] = out;
}

__global__ void dequantBiasKernel(
    int32_t const* acc, float const* weightScale, float const* bias, float* out, float activationScale)
{
    int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= kM * kN)
    {
        return;
    }
    int32_t n = idx % kN;
    out[idx] = static_cast<float>(acc[idx]) * activationScale * weightScale[n] + bias[n];
}

__global__ void dequantBiasVec4Kernel(
    int32_t const* acc, float const* weightScale, float const* bias, float* out, float activationScale)
{
    int32_t packIdx = blockIdx.x * blockDim.x + threadIdx.x;
    constexpr int32_t packs = (kM * kN) / 4;
    if (packIdx >= packs)
    {
        return;
    }
    int32_t base = packIdx * 4;
    int32_t n = base % kN;
    int4 av = reinterpret_cast<int4 const*>(acc)[packIdx];
    float4 sv = *reinterpret_cast<float4 const*>(weightScale + n);
    float4 bv = *reinterpret_cast<float4 const*>(bias + n);
    float4 ov{};
    ov.x = static_cast<float>(av.x) * activationScale * sv.x + bv.x;
    ov.y = static_cast<float>(av.y) * activationScale * sv.y + bv.y;
    ov.z = static_cast<float>(av.z) * activationScale * sv.z + bv.z;
    ov.w = static_cast<float>(av.w) * activationScale * sv.w + bv.w;
    reinterpret_cast<float4*>(out)[packIdx] = ov;
}

struct LtObjects
{
    cublasLtHandle_t handle{};
    cublasLtMatmulDesc_t opDesc{};
    cublasLtMatrixLayout_t aDesc{};
    cublasLtMatrixLayout_t bDesc{};
    cublasLtMatrixLayout_t cDesc{};
    cublasLtMatrixLayout_t dDesc{};
    cublasLtMatmulPreference_t preference{};
    cublasLtMatmulHeuristicResult_t heuristic{};
    bool hasAlgo{false};

    ~LtObjects()
    {
        if (preference)
        {
            cublasLtMatmulPreferenceDestroy(preference);
        }
        if (dDesc)
        {
            cublasLtMatrixLayoutDestroy(dDesc);
        }
        if (cDesc)
        {
            cublasLtMatrixLayoutDestroy(cDesc);
        }
        if (bDesc)
        {
            cublasLtMatrixLayoutDestroy(bDesc);
        }
        if (aDesc)
        {
            cublasLtMatrixLayoutDestroy(aDesc);
        }
        if (opDesc)
        {
            cublasLtMatmulDescDestroy(opDesc);
        }
        if (handle)
        {
            cublasLtDestroy(handle);
        }
    }
};

LtObjects makeLtObjects(void* workspace)
{
    LtObjects lt;
    checkCublas(cublasLtCreate(&lt.handle), "cublasLtCreate");
    checkCublas(cublasLtMatmulDescCreate(&lt.opDesc, CUBLAS_COMPUTE_32I, CUDA_R_32I), "cublasLtMatmulDescCreate");
    cublasOperation_t transA = CUBLAS_OP_N;
    cublasOperation_t transB = CUBLAS_OP_T;
    checkCublas(cublasLtMatmulDescSetAttribute(lt.opDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transA, sizeof(transA)),
        "set transA");
    checkCublas(cublasLtMatmulDescSetAttribute(lt.opDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transB, sizeof(transB)),
        "set transB");

    checkCublas(cublasLtMatrixLayoutCreate(&lt.aDesc, CUDA_R_8I, kM, kK, kK), "create aDesc");
    checkCublas(cublasLtMatrixLayoutCreate(&lt.bDesc, CUDA_R_8I, kN, kK, kK), "create bDesc");
    checkCublas(cublasLtMatrixLayoutCreate(&lt.cDesc, CUDA_R_32I, kM, kN, kN), "create cDesc");
    checkCublas(cublasLtMatrixLayoutCreate(&lt.dDesc, CUDA_R_32I, kM, kN, kN), "create dDesc");
    cublasLtOrder_t rowMajor = CUBLASLT_ORDER_ROW;
    checkCublas(cublasLtMatrixLayoutSetAttribute(lt.aDesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &rowMajor, sizeof(rowMajor)),
        "set a row major");
    checkCublas(cublasLtMatrixLayoutSetAttribute(lt.bDesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &rowMajor, sizeof(rowMajor)),
        "set b row major");
    checkCublas(cublasLtMatrixLayoutSetAttribute(lt.cDesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &rowMajor, sizeof(rowMajor)),
        "set c row major");
    checkCublas(cublasLtMatrixLayoutSetAttribute(lt.dDesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &rowMajor, sizeof(rowMajor)),
        "set d row major");

    checkCublas(cublasLtMatmulPreferenceCreate(&lt.preference), "cublasLtMatmulPreferenceCreate");
    checkCublas(cublasLtMatmulPreferenceSetAttribute(
                    lt.preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &kWorkspaceBytes, sizeof(kWorkspaceBytes)),
        "set workspace preference");

    int32_t returned = 0;
    checkCublas(cublasLtMatmulAlgoGetHeuristic(lt.handle, lt.opDesc, lt.aDesc, lt.bDesc, lt.cDesc, lt.dDesc,
                    lt.preference, 1, &lt.heuristic, &returned),
        "cublasLtMatmulAlgoGetHeuristic");
    lt.hasAlgo = returned > 0 && lt.heuristic.state == CUBLAS_STATUS_SUCCESS;
    if (!lt.hasAlgo)
    {
        std::fprintf(stderr, "no cublasLt int8 matmul algorithm found\n");
        std::exit(1);
    }
    (void) workspace;
    return lt;
}

float* gX{};
float* gSmooth{};
int8_t* gQx{};
int8_t* gQw{};
float* gWScale{};
float* gBias{};
int32_t* gAcc{};
float* gOut{};
float gActivationScale{};
void* gWorkspace{};
LtObjects* gLt{};

void launchQuantize(cudaStream_t stream)
{
    int32_t total = kM * kK;
    smoothQuantActivationKernel<<<(total + 255) / 256, 256, 0, stream>>>(gX, gSmooth, gQx, gActivationScale, total);
}

void launchQuantizeVec4(cudaStream_t stream)
{
    int32_t packs = (kM * kK) / 4;
    smoothQuantActivationVec4Kernel<<<(packs + 255) / 256, 256, 0, stream>>>(gX, gSmooth, gQx, gActivationScale);
}

void launchCublasLtGemm(cudaStream_t stream)
{
    int32_t alpha = 1;
    int32_t beta = 0;
    checkCublas(cublasLtMatmul(gLt->handle, gLt->opDesc, &alpha, gQx, gLt->aDesc, gQw, gLt->bDesc, &beta, gAcc,
                    gLt->cDesc, gAcc, gLt->dDesc, &gLt->heuristic.algo, gWorkspace, kWorkspaceBytes, stream),
        "cublasLtMatmul");
}

void launchDequant(cudaStream_t stream)
{
    dequantBiasKernel<<<(kM * kN + 255) / 256, 256, 0, stream>>>(gAcc, gWScale, gBias, gOut, gActivationScale);
}

void launchDequantVec4(cudaStream_t stream)
{
    int32_t packs = (kM * kN) / 4;
    dequantBiasVec4Kernel<<<(packs + 255) / 256, 256, 0, stream>>>(gAcc, gWScale, gBias, gOut, gActivationScale);
}

void launchCublasLtFull(cudaStream_t stream)
{
    launchCublasLtGemm(stream);
    launchDequant(stream);
}

void launchCublasLtFullVec4(cudaStream_t stream)
{
    launchCublasLtGemm(stream);
    launchDequantVec4(stream);
}

void launchCublasLtTotalVec4(cudaStream_t stream)
{
    launchQuantizeVec4(stream);
    launchCublasLtGemm(stream);
    launchDequantVec4(stream);
}

float timeCuda(cudaStream_t stream, int32_t warmup, int32_t iters, void (*fn)(cudaStream_t))
{
    for (int32_t i = 0; i < warmup; ++i)
    {
        fn(stream);
    }
    checkCuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize warmup");
    cudaEvent_t start{}, stop{};
    checkCuda(cudaEventCreate(&start), "cudaEventCreate start");
    checkCuda(cudaEventCreate(&stop), "cudaEventCreate stop");
    checkCuda(cudaEventRecord(start, stream), "cudaEventRecord start");
    for (int32_t i = 0; i < iters; ++i)
    {
        fn(stream);
    }
    checkCuda(cudaEventRecord(stop, stream), "cudaEventRecord stop");
    checkCuda(cudaEventSynchronize(stop), "cudaEventSynchronize stop");
    float totalMs = 0.0F;
    checkCuda(cudaEventElapsedTime(&totalMs, start, stop), "cudaEventElapsedTime");
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return totalMs / static_cast<float>(std::max(1, iters));
}

} // namespace

int main(int argc, char** argv)
{
    std::filesystem::path dataDir = argc > 1 ? argv[1] : "runs/deploy/6/G-smoothquant-linear-epilogue/tensors";
    std::filesystem::path outDir = argc > 2 ? argv[2] : "runs/deploy/6/G-smoothquant-linear-epilogue";
    gActivationScale = argc > 3 ? std::atof(argv[3]) : 1.0F;
    int32_t warmup = argc > 4 ? std::atoi(argv[4]) : 20;
    int32_t iters = argc > 5 ? std::atoi(argv[5]) : 200;
    std::filesystem::create_directories(outDir);

    auto hX = readFloatFile(dataDir / "x.fp32.bin", static_cast<size_t>(kM) * kK);
    auto hSmooth = readFloatFile(dataDir / "smooth_scale.fp32.bin", kK);
    auto hQw = readInt8File(dataDir / "qweight.int8.bin", static_cast<size_t>(kN) * kK);
    auto hWScale = readFloatFile(dataDir / "weight_scale.fp32.bin", kN);
    auto hBias = readFloatFile(dataDir / "bias.fp32.bin", kN);
    auto hRef = readFloatFile(dataDir / "reference.fp32.bin", static_cast<size_t>(kM) * kN);

    checkCuda(cudaMalloc(&gX, hX.size() * sizeof(float)), "cudaMalloc x");
    checkCuda(cudaMalloc(&gSmooth, hSmooth.size() * sizeof(float)), "cudaMalloc smooth");
    checkCuda(cudaMalloc(&gQx, static_cast<size_t>(kM) * kK * sizeof(int8_t)), "cudaMalloc qx");
    checkCuda(cudaMalloc(&gQw, hQw.size() * sizeof(int8_t)), "cudaMalloc qw");
    checkCuda(cudaMalloc(&gWScale, hWScale.size() * sizeof(float)), "cudaMalloc wscale");
    checkCuda(cudaMalloc(&gBias, hBias.size() * sizeof(float)), "cudaMalloc bias");
    checkCuda(cudaMalloc(&gAcc, static_cast<size_t>(kM) * kN * sizeof(int32_t)), "cudaMalloc acc");
    checkCuda(cudaMalloc(&gOut, static_cast<size_t>(kM) * kN * sizeof(float)), "cudaMalloc out");
    checkCuda(cudaMalloc(&gWorkspace, kWorkspaceBytes), "cudaMalloc workspace");
    checkCuda(cudaMemcpy(gX, hX.data(), hX.size() * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy x");
    checkCuda(cudaMemcpy(gSmooth, hSmooth.data(), hSmooth.size() * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy smooth");
    checkCuda(cudaMemcpy(gQw, hQw.data(), hQw.size() * sizeof(int8_t), cudaMemcpyHostToDevice), "cudaMemcpy qw");
    checkCuda(cudaMemcpy(gWScale, hWScale.data(), hWScale.size() * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy wscale");
    checkCuda(cudaMemcpy(gBias, hBias.data(), hBias.size() * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy bias");

    cudaStream_t stream{};
    checkCuda(cudaStreamCreate(&stream), "cudaStreamCreate");
    LtObjects lt = makeLtObjects(gWorkspace);
    gLt = &lt;

    float quantizeMs = timeCuda(stream, warmup, iters, launchQuantize);
    float quantizeVec4Ms = timeCuda(stream, warmup, iters, launchQuantizeVec4);
    float gemmMs = timeCuda(stream, warmup, iters, launchCublasLtGemm);
    float dequantMs = timeCuda(stream, warmup, iters, launchDequant);
    float dequantVec4Ms = timeCuda(stream, warmup, iters, launchDequantVec4);
    float totalMs = timeCuda(stream, warmup, iters, launchCublasLtFull);
    float totalVec4Ms = timeCuda(stream, warmup, iters, launchCublasLtFullVec4);
    float totalWithQuantizeVec4Ms = timeCuda(stream, warmup, iters, launchCublasLtTotalVec4);

    std::vector<float> hOut(static_cast<size_t>(kM) * kN);
    checkCuda(cudaMemcpy(hOut.data(), gOut, hOut.size() * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy out");
    writeFloatFile(outDir / "output_cublaslt.fp32.bin", hOut);
    Metrics metrics = computeMetrics(hRef, hOut);

    std::filesystem::path reportPath = outDir / "smoothquant_linear_cublaslt_report.json";
    std::ofstream report(reportPath);
    report << "{\n"
           << "  \"target\": \"layer0.self_attn.q_proj\",\n"
           << "  \"m\": " << kM << ",\n"
           << "  \"k\": " << kK << ",\n"
           << "  \"n\": " << kN << ",\n"
           << "  \"activation_scale\": " << gActivationScale << ",\n"
           << "  \"warmup\": " << warmup << ",\n"
           << "  \"iters\": " << iters << ",\n"
           << "  \"latency_ms\": {\n"
           << "    \"smoothquant_activation_quantize\": " << quantizeMs << ",\n"
           << "    \"smoothquant_activation_quantize_vec4\": " << quantizeVec4Ms << ",\n"
           << "    \"cublaslt_int8_gemm_int32\": " << gemmMs << ",\n"
           << "    \"dequant_bias_kernel\": " << dequantMs << ",\n"
           << "    \"dequant_bias_vec4_kernel\": " << dequantVec4Ms << ",\n"
           << "    \"cublaslt_gemm_plus_dequant_bias\": " << totalMs << ",\n"
           << "    \"cublaslt_gemm_plus_dequant_bias_vec4\": " << totalVec4Ms << ",\n"
           << "    \"total_with_activation_quantize\": " << (quantizeMs + totalMs) << ",\n"
           << "    \"total_with_activation_quantize_vec4\": " << totalWithQuantizeVec4Ms << "\n"
           << "  },\n"
           << "  \"vectorized_speedup\": {\n"
           << "    \"activation_quantize\": " << (quantizeMs / std::max(quantizeVec4Ms, 1.0e-12F)) << ",\n"
           << "    \"dequant_bias\": " << (dequantMs / std::max(dequantVec4Ms, 1.0e-12F)) << ",\n"
           << "    \"gemm_plus_dequant_bias\": " << (totalMs / std::max(totalVec4Ms, 1.0e-12F)) << ",\n"
           << "    \"total_with_activation_quantize\": " << ((quantizeMs + totalMs) / std::max(totalWithQuantizeVec4Ms, 1.0e-12F)) << "\n"
           << "  },\n"
           << "  \"metrics\": {\n"
           << "    \"cosine\": " << metrics.cosine << ",\n"
           << "    \"relative_l2\": " << metrics.relativeL2 << ",\n"
           << "    \"max_abs\": " << metrics.maxAbs << "\n"
           << "  }\n"
           << "}\n";
    report.close();

    std::printf("SmoothQuant activation quantize: %.6f ms\n", quantizeMs);
    std::printf("SmoothQuant activation quantize vec4: %.6f ms\n", quantizeVec4Ms);
    std::printf("cublasLt INT8 GEMM int32: %.6f ms\n", gemmMs);
    std::printf("dequant+bias kernel: %.6f ms\n", dequantMs);
    std::printf("dequant+bias vec4 kernel: %.6f ms\n", dequantVec4Ms);
    std::printf("cublasLt GEMM + dequant/bias: %.6f ms\n", totalMs);
    std::printf("cublasLt GEMM + dequant/bias vec4: %.6f ms\n", totalVec4Ms);
    std::printf("total with activation quantize: %.6f ms\n", quantizeMs + totalMs);
    std::printf("total with activation quantize vec4: %.6f ms\n", totalWithQuantizeVec4Ms);
    std::printf("relative_l2: %.9g max_abs: %.9g\n", metrics.relativeL2, metrics.maxAbs);

    cudaStreamDestroy(stream);
    cudaFree(gX);
    cudaFree(gSmooth);
    cudaFree(gQx);
    cudaFree(gQw);
    cudaFree(gWScale);
    cudaFree(gBias);
    cudaFree(gAcc);
    cudaFree(gOut);
    cudaFree(gWorkspace);
    return 0;
}
