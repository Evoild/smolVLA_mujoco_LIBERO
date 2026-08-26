#include <NvInfer.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

using namespace nvinfer1;

namespace
{

constexpr int32_t kBatch = 1;
constexpr int32_t kSeqLen = 177;
constexpr int32_t kQHeads = 15;
constexpr int32_t kKvHeads = 5;
constexpr int32_t kHeadDim = 64;
constexpr int32_t kHalfDim = kHeadDim / 2;
constexpr int32_t kGroups = kQHeads / kKvHeads;

class Logger final : public ILogger
{
public:
    void log(Severity severity, AsciiChar const* msg) noexcept override
    {
        if (severity <= Severity::kINFO)
        {
            std::fprintf(stderr, "[TRT] %s\n", msg);
        }
    }
};

template <typename T>
struct TrtDeleter
{
    void operator()(T* p) const { delete p; }
};

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

struct ConstantStore
{
    std::vector<std::vector<float>> floats;

    Weights add(std::vector<float> values)
    {
        floats.push_back(std::move(values));
        return Weights{DataType::kFLOAT, floats.back().data(), static_cast<int64_t>(floats.back().size())};
    }
};

ITensor* addConstant(INetworkDefinition& network, ConstantStore& store, Dims dims, std::vector<float> values, char const* name)
{
    IConstantLayer* layer = network.addConstant(dims, store.add(std::move(values)));
    if (!layer)
    {
        std::fprintf(stderr, "addConstant failed: %s\n", name);
        std::exit(1);
    }
    layer->setName(name);
    return layer->getOutput(0);
}

ITensor* addElementwise(INetworkDefinition& network, ITensor& a, ITensor& b, ElementWiseOperation op, char const* name)
{
    IElementWiseLayer* layer = network.addElementWise(a, b, op);
    if (!layer)
    {
        std::fprintf(stderr, "addElementWise failed: %s\n", name);
        std::exit(1);
    }
    layer->setName(name);
    return layer->getOutput(0);
}

ITensor* addSlice(INetworkDefinition& network, ITensor& input, Dims start, Dims size, Dims stride, char const* name)
{
    ISliceLayer* layer = network.addSlice(input, start, size, stride);
    if (!layer)
    {
        std::fprintf(stderr, "addSlice failed: %s\n", name);
        std::exit(1);
    }
    layer->setName(name);
    return layer->getOutput(0);
}

ITensor* addConcat(INetworkDefinition& network, std::vector<ITensor*> const& tensors, int32_t axis, char const* name)
{
    IConcatenationLayer* layer = network.addConcatenation(tensors.data(), static_cast<int32_t>(tensors.size()));
    if (!layer)
    {
        std::fprintf(stderr, "addConcatenation failed: %s\n", name);
        std::exit(1);
    }
    layer->setAxis(axis);
    layer->setName(name);
    return layer->getOutput(0);
}

ITensor* addShuffle(INetworkDefinition& network, ITensor& input, Permutation perm, char const* name)
{
    IShuffleLayer* layer = network.addShuffle(input);
    if (!layer)
    {
        std::fprintf(stderr, "addShuffle failed: %s\n", name);
        std::exit(1);
    }
    layer->setFirstTranspose(perm);
    layer->setName(name);
    return layer->getOutput(0);
}

ITensor* addMatMul(INetworkDefinition& network, ITensor& a, ITensor& b, char const* name)
{
    IMatrixMultiplyLayer* layer = network.addMatrixMultiply(a, MatrixOperation::kNONE, b, MatrixOperation::kNONE);
    if (!layer)
    {
        std::fprintf(stderr, "addMatrixMultiply failed: %s\n", name);
        std::exit(1);
    }
    layer->setName(name);
    return layer->getOutput(0);
}

std::vector<float> makeTrig(std::vector<int64_t> const& positionIds, bool sine)
{
    std::vector<float> values(static_cast<size_t>(kBatch) * kSeqLen * 1 * kHalfDim);
    for (int32_t b = 0; b < kBatch; ++b)
    {
        for (int32_t pos = 0; pos < kSeqLen; ++pos)
        {
            int64_t position = positionIds[static_cast<size_t>(b) * kSeqLen + pos];
            for (int32_t d = 0; d < kHalfDim; ++d)
            {
                float exponent = (2.0F / static_cast<float>(kHeadDim)) * static_cast<float>(d);
                float invFreq = 1.0F / std::pow(10000.0F, exponent);
                float radians = static_cast<float>(position) * invFreq;
                values[(static_cast<size_t>(b) * kSeqLen + pos) * kHalfDim + d] = sine ? std::sin(radians) : std::cos(radians);
            }
        }
    }
    return values;
}

ITensor* addRope(INetworkDefinition& network, ConstantStore& store, ITensor& raw, int32_t heads, ITensor& cosHalf, ITensor& sinHalf,
    char const* prefix)
{
    Dims4 firstStart{0, 0, 0, 0};
    Dims4 firstSize{kBatch, kSeqLen, heads, kHalfDim};
    Dims4 secondStart{0, 0, 0, kHalfDim};
    Dims4 stride{1, 1, 1, 1};
    ITensor* x1 = addSlice(network, raw, firstStart, firstSize, stride, (std::string(prefix) + "_slice_first").c_str());
    ITensor* x2 = addSlice(network, raw, secondStart, firstSize, stride, (std::string(prefix) + "_slice_second").c_str());
    ITensor* x1Cos = addElementwise(network, *x1, cosHalf, ElementWiseOperation::kPROD, (std::string(prefix) + "_x1_cos").c_str());
    ITensor* x2Sin = addElementwise(network, *x2, sinHalf, ElementWiseOperation::kPROD, (std::string(prefix) + "_x2_sin").c_str());
    ITensor* rotFirst = addElementwise(network, *x1Cos, *x2Sin, ElementWiseOperation::kSUB, (std::string(prefix) + "_rot_first").c_str());
    ITensor* x2Cos = addElementwise(network, *x2, cosHalf, ElementWiseOperation::kPROD, (std::string(prefix) + "_x2_cos").c_str());
    ITensor* x1Sin = addElementwise(network, *x1, sinHalf, ElementWiseOperation::kPROD, (std::string(prefix) + "_x1_sin").c_str());
    ITensor* rotSecond = addElementwise(network, *x2Cos, *x1Sin, ElementWiseOperation::kSUM, (std::string(prefix) + "_rot_second").c_str());
    return addConcat(network, {rotFirst, rotSecond}, 3, (std::string(prefix) + "_rope").c_str());
}

ITensor* addRepeatKvHeads(INetworkDefinition& network, ITensor& input, char const* prefix)
{
    std::vector<ITensor*> repeated;
    Dims4 stride{1, 1, 1, 1};
    for (int32_t kvh = 0; kvh < kKvHeads; ++kvh)
    {
        ITensor* one = addSlice(network, input, Dims4{0, 0, kvh, 0}, Dims4{kBatch, kSeqLen, 1, kHeadDim}, stride,
            (std::string(prefix) + "_slice_" + std::to_string(kvh)).c_str());
        for (int32_t g = 0; g < kGroups; ++g)
        {
            repeated.push_back(one);
        }
    }
    return addConcat(network, repeated, 2, (std::string(prefix) + "_repeat_heads").c_str());
}

} // namespace

int main(int argc, char** argv)
{
    std::filesystem::path dataDir = argc > 1 ? argv[1] : "runs/deploy/6/F-layer0-real-attention/tensors";
    std::filesystem::path outDir = argc > 2 ? argv[2] : "runs/deploy/6/F-layer0-real-attention";
    int32_t warmup = argc > 3 ? std::atoi(argv[3]) : 20;
    int32_t iters = argc > 4 ? std::atoi(argv[4]) : 200;
    std::filesystem::create_directories(outDir);

    Logger logger;
    std::unique_ptr<IBuilder, TrtDeleter<IBuilder>> builder{createInferBuilder(logger)};
    std::unique_ptr<INetworkDefinition, TrtDeleter<INetworkDefinition>> network{builder ? builder->createNetworkV2(0U) : nullptr};
    std::unique_ptr<IBuilderConfig, TrtDeleter<IBuilderConfig>> config{builder ? builder->createBuilderConfig() : nullptr};
    if (!builder || !network || !config)
    {
        std::fprintf(stderr, "create builder/network/config failed\n");
        return 1;
    }

    size_t qCount = static_cast<size_t>(kBatch) * kSeqLen * kQHeads * kHeadDim;
    size_t kvCount = static_cast<size_t>(kBatch) * kSeqLen * kKvHeads * kHeadDim;
    size_t posCount = static_cast<size_t>(kBatch) * kSeqLen;
    size_t maskCount = static_cast<size_t>(kBatch) * kSeqLen * kSeqLen;
    size_t outCount = qCount;

    auto hQ = readFloatFile(dataDir / "q_raw.fp32.bin", qCount);
    auto hK = readFloatFile(dataDir / "k_raw.fp32.bin", kvCount);
    auto hV = readFloatFile(dataDir / "v_raw.fp32.bin", kvCount);
    auto hPosBytes = readBytes(dataDir / "position_ids.int64.bin", posCount * sizeof(int64_t));
    auto hMaskBytes = readBytes(dataDir / "attention_mask.bool.bin", maskCount * sizeof(bool));
    auto hRef = readFloatFile(dataDir / "context_ref.fp32.bin", outCount);
    std::vector<int64_t> hPos(posCount);
    std::memcpy(hPos.data(), hPosBytes.data(), hPosBytes.size());

    TensorFormats linear = 1U << static_cast<uint32_t>(TensorFormat::kLINEAR);
    ITensor* q = network->addInput("q_raw", DataType::kFLOAT, Dims4{kBatch, kSeqLen, kQHeads, kHeadDim});
    ITensor* k = network->addInput("k_raw", DataType::kFLOAT, Dims4{kBatch, kSeqLen, kKvHeads, kHeadDim});
    ITensor* v = network->addInput("v_raw", DataType::kFLOAT, Dims4{kBatch, kSeqLen, kKvHeads, kHeadDim});
    ITensor* mask = network->addInput("attention_mask", DataType::kBOOL, Dims4{kBatch, 1, kSeqLen, kSeqLen});
    for (ITensor* t : {q, k, v, mask})
    {
        if (!t)
        {
            std::fprintf(stderr, "addInput failed\n");
            return 1;
        }
        t->setAllowedFormats(linear);
    }

    ConstantStore store;
    ITensor* cosHalf = addConstant(*network, store, Dims4{kBatch, kSeqLen, 1, kHalfDim}, makeTrig(hPos, false), "cos_half");
    ITensor* sinHalf = addConstant(*network, store, Dims4{kBatch, kSeqLen, 1, kHalfDim}, makeTrig(hPos, true), "sin_half");

    ITensor* qRope = addRope(*network, store, *q, kQHeads, *cosHalf, *sinHalf, "q");
    ITensor* kRope = addRope(*network, store, *k, kKvHeads, *cosHalf, *sinHalf, "k");
    ITensor* kExpanded = addRepeatKvHeads(*network, *kRope, "k");
    ITensor* vExpanded = addRepeatKvHeads(*network, *v, "v");
    ITensor* qT = addShuffle(*network, *qRope, Permutation{0, 2, 1, 3}, "q_transpose");
    ITensor* kT = addShuffle(*network, *kExpanded, Permutation{0, 2, 3, 1}, "k_transpose");
    ITensor* vT = addShuffle(*network, *vExpanded, Permutation{0, 2, 1, 3}, "v_transpose");

    ITensor* scoresRaw = addMatMul(*network, *qT, *kT, "qk_matmul");
    ITensor* scale = addConstant(*network, store, Dims4{1, 1, 1, 1}, {1.0F / std::sqrt(static_cast<float>(kHeadDim))}, "score_scale");
    ITensor* scores = addElementwise(*network, *scoresRaw, *scale, ElementWiseOperation::kPROD, "scores_scale");
    ITensor* negLarge = addConstant(*network, store, Dims4{1, 1, 1, 1}, {-3.4028234663852886e38F}, "mask_fill_value");
    ISelectLayer* select = network->addSelect(*mask, *scores, *negLarge);
    if (!select)
    {
        std::fprintf(stderr, "addSelect failed\n");
        return 1;
    }
    select->setName("mask_select");
    ISoftMaxLayer* softmax = network->addSoftMax(*select->getOutput(0));
    if (!softmax)
    {
        std::fprintf(stderr, "addSoftMax failed\n");
        return 1;
    }
    softmax->setAxes(1U << 3);
    softmax->setName("softmax");
    ITensor* contextOut = addMatMul(*network, *softmax->getOutput(0), *vT, "pv_matmul");
    contextOut->setName("context");
    contextOut->setAllowedFormats(linear);
    network->markOutput(*contextOut);

    std::unique_ptr<IHostMemory, TrtDeleter<IHostMemory>> plan{builder->buildSerializedNetwork(*network, *config)};
    if (!plan || plan->size() == 0)
    {
        std::fprintf(stderr, "buildSerializedNetwork failed or returned empty plan\n");
        return 1;
    }
    std::filesystem::path planPath = outDir / "unfused_attention_trt_real.plan";
    std::ofstream planOs(planPath, std::ios::binary);
    planOs.write(static_cast<char const*>(plan->data()), static_cast<std::streamsize>(plan->size()));
    planOs.close();

    std::unique_ptr<IRuntime, TrtDeleter<IRuntime>> runtime{createInferRuntime(logger)};
    std::unique_ptr<ICudaEngine, TrtDeleter<ICudaEngine>> engine{
        runtime ? runtime->deserializeCudaEngine(plan->data(), plan->size()) : nullptr};
    std::unique_ptr<IExecutionContext, TrtDeleter<IExecutionContext>> context{engine ? engine->createExecutionContext() : nullptr};
    if (!runtime || !engine || !context)
    {
        std::fprintf(stderr, "runtime/engine/context creation failed\n");
        return 1;
    }

    void *dq{}, *dk{}, *dv{}, *dm{}, *do_{};
    checkCuda(cudaMalloc(&dq, qCount * sizeof(float)), "cudaMalloc q");
    checkCuda(cudaMalloc(&dk, kvCount * sizeof(float)), "cudaMalloc k");
    checkCuda(cudaMalloc(&dv, kvCount * sizeof(float)), "cudaMalloc v");
    checkCuda(cudaMalloc(&dm, maskCount * sizeof(bool)), "cudaMalloc mask");
    checkCuda(cudaMalloc(&do_, outCount * sizeof(float)), "cudaMalloc out");
    checkCuda(cudaMemcpy(dq, hQ.data(), qCount * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy q");
    checkCuda(cudaMemcpy(dk, hK.data(), kvCount * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy k");
    checkCuda(cudaMemcpy(dv, hV.data(), kvCount * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy v");
    checkCuda(cudaMemcpy(dm, hMaskBytes.data(), hMaskBytes.size(), cudaMemcpyHostToDevice), "cudaMemcpy mask");

    context->setTensorAddress("q_raw", dq);
    context->setTensorAddress("k_raw", dk);
    context->setTensorAddress("v_raw", dv);
    context->setTensorAddress("attention_mask", dm);
    context->setTensorAddress("context", do_);

    cudaStream_t stream{};
    checkCuda(cudaStreamCreate(&stream), "cudaStreamCreate");
    for (int32_t i = 0; i < warmup; ++i)
    {
        if (!context->enqueueV3(stream))
        {
            std::fprintf(stderr, "enqueueV3 failed during warmup\n");
            return 1;
        }
    }
    checkCuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize warmup");

    cudaEvent_t start{}, stop{};
    checkCuda(cudaEventCreate(&start), "cudaEventCreate start");
    checkCuda(cudaEventCreate(&stop), "cudaEventCreate stop");
    checkCuda(cudaEventRecord(start, stream), "cudaEventRecord start");
    for (int32_t i = 0; i < iters; ++i)
    {
        if (!context->enqueueV3(stream))
        {
            std::fprintf(stderr, "enqueueV3 failed during timing\n");
            return 1;
        }
    }
    checkCuda(cudaEventRecord(stop, stream), "cudaEventRecord stop");
    checkCuda(cudaEventSynchronize(stop), "cudaEventSynchronize stop");
    float totalMs = 0.0F;
    checkCuda(cudaEventElapsedTime(&totalMs, start, stop), "cudaEventElapsedTime");

    std::vector<float> hOut(outCount);
    checkCuda(cudaMemcpy(hOut.data(), do_, outCount * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy out");
    writeFloatFile(outDir / "context_unfused_trt.fp32.bin", hOut);
    Metrics m = computeMetrics(hRef, hOut);
    double latencyMs = static_cast<double>(totalMs) / static_cast<double>(std::max(1, iters));

    std::filesystem::path reportPath = outDir / "unfused_attention_trt_real_report.json";
    std::ofstream report(reportPath);
    report << "{\n"
           << "  \"plan\": \"" << planPath.string() << "\",\n"
           << "  \"plan_size_bytes\": " << plan->size() << ",\n"
           << "  \"batch\": " << kBatch << ",\n"
           << "  \"seq_len\": " << kSeqLen << ",\n"
           << "  \"q_heads\": " << kQHeads << ",\n"
           << "  \"kv_heads\": " << kKvHeads << ",\n"
           << "  \"head_dim\": " << kHeadDim << ",\n"
           << "  \"warmup\": " << warmup << ",\n"
           << "  \"iters\": " << iters << ",\n"
           << "  \"unfused_trt_latency_ms\": " << latencyMs << ",\n"
           << "  \"context_metrics\": {\n"
           << "    \"cosine\": " << m.cosine << ",\n"
           << "    \"relative_l2\": " << m.relativeL2 << ",\n"
           << "    \"max_abs\": " << m.maxAbs << "\n"
           << "  }\n"
           << "}\n";
    report.close();

    std::printf("built plan: %s (%zu bytes)\n", planPath.c_str(), plan->size());
    std::printf("enqueueV3 ok\n");
    std::printf("unfused TRT latency: %.6f ms\n", latencyMs);
    std::printf("context cosine: %.9f relative_l2: %.9g max_abs: %.9g\n", m.cosine, m.relativeL2, m.maxAbs);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaStreamDestroy(stream);
    cudaFree(dq);
    cudaFree(dk);
    cudaFree(dv);
    cudaFree(dm);
    cudaFree(do_);
    return 0;
}
