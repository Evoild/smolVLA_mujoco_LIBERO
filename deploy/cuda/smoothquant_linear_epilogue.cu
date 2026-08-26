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
constexpr int32_t kTile = 16;

void checkCuda(cudaError_t err, char const* what)
{
    if (err != cudaSuccess)
    {
        std::fprintf(stderr, "%s failed: %s\n", what, cudaGetErrorString(err));
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

__global__ void int8GemmInt32Kernel(int8_t const* qx, int8_t const* qw, int32_t* accOut)
{
    int32_t n = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t m = blockIdx.y * blockDim.y + threadIdx.y;
    if (m >= kM || n >= kN)
    {
        return;
    }
    int32_t acc = 0;
    for (int32_t k = 0; k < kK; ++k)
    {
        int32_t a = static_cast<int32_t>(qx[m * kK + k]);
        int32_t b = static_cast<int32_t>(qw[n * kK + k]);
        acc += a * b;
    }
    accOut[m * kN + n] = acc;
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

__global__ void int8GemmFusedEpilogueKernel(
    int8_t const* qx, int8_t const* qw, float const* weightScale, float const* bias, float* out, float activationScale)
{
    int32_t n = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t m = blockIdx.y * blockDim.y + threadIdx.y;
    if (m >= kM || n >= kN)
    {
        return;
    }
    int32_t acc = 0;
    for (int32_t k = 0; k < kK; ++k)
    {
        int32_t a = static_cast<int32_t>(qx[m * kK + k]);
        int32_t b = static_cast<int32_t>(qw[n * kK + k]);
        acc += a * b;
    }
    out[m * kN + n] = static_cast<float>(acc) * activationScale * weightScale[n] + bias[n];
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

float* gX{};
float* gSmooth{};
int8_t* gQx{};
int8_t* gQw{};
float* gWScale{};
float* gBias{};
int32_t* gAcc{};
float* gOutUnfused{};
float* gOutFused{};
float gActivationScale{};

void launchQuantize(cudaStream_t stream)
{
    int32_t total = kM * kK;
    smoothQuantActivationKernel<<<(total + 255) / 256, 256, 0, stream>>>(gX, gSmooth, gQx, gActivationScale, total);
}

void launchUnfused(cudaStream_t stream)
{
    dim3 block(kTile, kTile);
    dim3 grid((kN + kTile - 1) / kTile, (kM + kTile - 1) / kTile);
    int8GemmInt32Kernel<<<grid, block, 0, stream>>>(gQx, gQw, gAcc);
    dequantBiasKernel<<<(kM * kN + 255) / 256, 256, 0, stream>>>(gAcc, gWScale, gBias, gOutUnfused, gActivationScale);
}

void launchFused(cudaStream_t stream)
{
    dim3 block(kTile, kTile);
    dim3 grid((kN + kTile - 1) / kTile, (kM + kTile - 1) / kTile);
    int8GemmFusedEpilogueKernel<<<grid, block, 0, stream>>>(gQx, gQw, gWScale, gBias, gOutFused, gActivationScale);
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
    checkCuda(cudaMalloc(&gOutUnfused, static_cast<size_t>(kM) * kN * sizeof(float)), "cudaMalloc out unfused");
    checkCuda(cudaMalloc(&gOutFused, static_cast<size_t>(kM) * kN * sizeof(float)), "cudaMalloc out fused");
    checkCuda(cudaMemcpy(gX, hX.data(), hX.size() * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy x");
    checkCuda(cudaMemcpy(gSmooth, hSmooth.data(), hSmooth.size() * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy smooth");
    checkCuda(cudaMemcpy(gQw, hQw.data(), hQw.size() * sizeof(int8_t), cudaMemcpyHostToDevice), "cudaMemcpy qw");
    checkCuda(cudaMemcpy(gWScale, hWScale.data(), hWScale.size() * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy wscale");
    checkCuda(cudaMemcpy(gBias, hBias.data(), hBias.size() * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy bias");

    cudaStream_t stream{};
    checkCuda(cudaStreamCreate(&stream), "cudaStreamCreate");
    float quantizeMs = timeCuda(stream, warmup, iters, launchQuantize);
    float unfusedMs = timeCuda(stream, warmup, iters, launchUnfused);
    float fusedMs = timeCuda(stream, warmup, iters, launchFused);

    std::vector<float> hOutUnfused(static_cast<size_t>(kM) * kN);
    std::vector<float> hOutFused(static_cast<size_t>(kM) * kN);
    checkCuda(cudaMemcpy(hOutUnfused.data(), gOutUnfused, hOutUnfused.size() * sizeof(float), cudaMemcpyDeviceToHost),
        "cudaMemcpy out unfused");
    checkCuda(cudaMemcpy(hOutFused.data(), gOutFused, hOutFused.size() * sizeof(float), cudaMemcpyDeviceToHost),
        "cudaMemcpy out fused");
    writeFloatFile(outDir / "output_unfused.fp32.bin", hOutUnfused);
    writeFloatFile(outDir / "output_fused_epilogue.fp32.bin", hOutFused);
    Metrics unfusedMetrics = computeMetrics(hRef, hOutUnfused);
    Metrics fusedMetrics = computeMetrics(hRef, hOutFused);

    std::filesystem::path reportPath = outDir / "smoothquant_linear_epilogue_report.json";
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
           << "    \"unfused_int8_gemm_int32_plus_dequant_bias\": " << unfusedMs << ",\n"
           << "    \"fused_int8_gemm_dequant_bias_epilogue\": " << fusedMs << ",\n"
           << "    \"unfused_total\": " << (quantizeMs + unfusedMs) << ",\n"
           << "    \"fused_total\": " << (quantizeMs + fusedMs) << "\n"
           << "  },\n"
           << "  \"fused_vs_unfused\": {\n"
           << "    \"epilogue_speedup\": " << (unfusedMs / std::max(fusedMs, 1.0e-12F)) << ",\n"
           << "    \"total_speedup\": " << ((quantizeMs + unfusedMs) / std::max(quantizeMs + fusedMs, 1.0e-12F)) << "\n"
           << "  },\n"
           << "  \"unfused_metrics\": {\n"
           << "    \"cosine\": " << unfusedMetrics.cosine << ",\n"
           << "    \"relative_l2\": " << unfusedMetrics.relativeL2 << ",\n"
           << "    \"max_abs\": " << unfusedMetrics.maxAbs << "\n"
           << "  },\n"
           << "  \"fused_metrics\": {\n"
           << "    \"cosine\": " << fusedMetrics.cosine << ",\n"
           << "    \"relative_l2\": " << fusedMetrics.relativeL2 << ",\n"
           << "    \"max_abs\": " << fusedMetrics.maxAbs << "\n"
           << "  }\n"
           << "}\n";
    report.close();

    std::printf("SmoothQuant activation quantize: %.6f ms\n", quantizeMs);
    std::printf("unfused int8 GEMM + dequant/bias: %.6f ms\n", unfusedMs);
    std::printf("fused epilogue int8 GEMM: %.6f ms\n", fusedMs);
    std::printf("fused relative_l2: %.9g max_abs: %.9g\n", fusedMetrics.relativeL2, fusedMetrics.maxAbs);

    cudaStreamDestroy(stream);
    cudaFree(gX);
    cudaFree(gSmooth);
    cudaFree(gQx);
    cudaFree(gQw);
    cudaFree(gWScale);
    cudaFree(gBias);
    cudaFree(gAcc);
    cudaFree(gOutUnfused);
    cudaFree(gOutFused);
    return 0;
}
