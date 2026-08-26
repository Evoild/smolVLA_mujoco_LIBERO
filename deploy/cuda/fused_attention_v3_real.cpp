#include "fused_rope_layout.h"

#include <NvInfer.h>
#include <NvInferPlugin.h>
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

class FusedAttentionRealV3 final
    : public IPluginV3
    , public IPluginV3OneCore
    , public IPluginV3OneBuild
    , public IPluginV3OneRuntime
{
public:
    explicit FusedAttentionRealV3(std::string name)
        : mName(std::move(name))
    {
        mParams.qHeads = kQHeads;
        mParams.kvHeads = kKvHeads;
        mParams.headDim = kHeadDim;
        mParams.maxWavelength = 10000.0F;
        mSerializeFields.nbFields = 0;
        mSerializeFields.fields = nullptr;
    }

    IPluginCapability* getCapabilityInterface(PluginCapabilityType type) noexcept override
    {
        if (type == PluginCapabilityType::kCORE)
        {
            return static_cast<IPluginV3OneCore*>(this);
        }
        if (type == PluginCapabilityType::kBUILD)
        {
            return static_cast<IPluginV3OneBuild*>(this);
        }
        if (type == PluginCapabilityType::kRUNTIME)
        {
            return static_cast<IPluginV3OneRuntime*>(this);
        }
        return nullptr;
    }

    IPluginV3* clone() noexcept override { return new FusedAttentionRealV3(mName); }

    AsciiChar const* getPluginName() const noexcept override { return "SmolVLAFusedRopeAttentionV3Real"; }
    AsciiChar const* getPluginVersion() const noexcept override { return "1"; }
    AsciiChar const* getPluginNamespace() const noexcept override { return ""; }

    int32_t configurePlugin(DynamicPluginTensorDesc const*, int32_t, DynamicPluginTensorDesc const*, int32_t) noexcept override
    {
        return 0;
    }

    int32_t getOutputDataTypes(DataType* outputTypes, int32_t nbOutputs, DataType const*, int32_t) const noexcept override
    {
        if (nbOutputs != 1)
        {
            return 1;
        }
        outputTypes[0] = DataType::kFLOAT;
        return 0;
    }

    int32_t getOutputShapes(DimsExprs const* inputs, int32_t nbInputs, DimsExprs const*, int32_t, DimsExprs* outputs,
        int32_t nbOutputs, IExprBuilder& exprBuilder) noexcept override
    {
        if (nbInputs != 5 || nbOutputs != 1)
        {
            return 1;
        }
        outputs[0].nbDims = 4;
        outputs[0].d[0] = inputs[0].d[0];
        outputs[0].d[1] = exprBuilder.constant(kQHeads);
        outputs[0].d[2] = inputs[0].d[1];
        outputs[0].d[3] = exprBuilder.constant(kHeadDim);
        return 0;
    }

    bool supportsFormatCombination(
        int32_t pos, DynamicPluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept override
    {
        if (nbInputs != 5 || nbOutputs != 1 || pos < 0 || pos >= 6)
        {
            return false;
        }
        if (inOut[pos].desc.format != TensorFormat::kLINEAR)
        {
            return false;
        }
        if (pos == 3)
        {
            return inOut[pos].desc.type == DataType::kINT64;
        }
        if (pos == 4)
        {
            return inOut[pos].desc.type == DataType::kBOOL;
        }
        return inOut[pos].desc.type == DataType::kFLOAT;
    }

    int32_t getNbOutputs() const noexcept override { return 1; }

    size_t getWorkspaceSize(DynamicPluginTensorDesc const*, int32_t, DynamicPluginTensorDesc const*, int32_t) const noexcept override
    {
        return 0;
    }

    int32_t getFormatCombinationLimit() noexcept override { return 1; }

    int32_t onShapeChange(PluginTensorDesc const*, int32_t, PluginTensorDesc const*, int32_t) noexcept override { return 0; }

    int32_t enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const*, void const* const* inputs,
        void* const* outputs, void*, cudaStream_t stream) noexcept override
    {
        int32_t batch = inputDesc[0].dims.d[0];
        int32_t seqLen = inputDesc[0].dims.d[1];
        return smolvla::launchFusedRopeAttention(inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], outputs[0],
            batch, seqLen, mParams, DataType::kFLOAT, DataType::kINT64, stream);
    }

    IPluginV3* attachToContext(IPluginResourceContext*) noexcept override { return clone(); }
    PluginFieldCollection const* getFieldsToSerialize() noexcept override { return &mSerializeFields; }

private:
    std::string mName;
    smolvla::FusedRopeLayoutParams mParams{};
    PluginFieldCollection mSerializeFields{};
};

class FusedAttentionRealV3Creator final : public IPluginCreatorV3One
{
public:
    FusedAttentionRealV3Creator()
    {
        mFields.nbFields = 0;
        mFields.fields = nullptr;
    }

    IPluginV3* createPlugin(AsciiChar const* name, PluginFieldCollection const*, TensorRTPhase) noexcept override
    {
        return new FusedAttentionRealV3(name ? name : "fused_attention_v3_real");
    }

    PluginFieldCollection const* getFieldNames() noexcept override { return &mFields; }
    AsciiChar const* getPluginName() const noexcept override { return "SmolVLAFusedRopeAttentionV3Real"; }
    AsciiChar const* getPluginVersion() const noexcept override { return "1"; }
    AsciiChar const* getPluginNamespace() const noexcept override { return ""; }

private:
    PluginFieldCollection mFields{};
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

} // namespace

int main(int argc, char** argv)
{
    std::filesystem::path dataDir = argc > 1 ? argv[1] : "runs/deploy/6/F-layer0-real-attention/tensors";
    std::filesystem::path outDir = argc > 2 ? argv[2] : "runs/deploy/6/F-layer0-real-attention";
    int32_t warmup = argc > 3 ? std::atoi(argv[3]) : 20;
    int32_t iters = argc > 4 ? std::atoi(argv[4]) : 200;
    std::filesystem::create_directories(outDir);

    FusedAttentionRealV3Creator creator;
    getPluginRegistry()->registerCreator(creator, "");
    Logger logger;
    std::unique_ptr<IBuilder, TrtDeleter<IBuilder>> builder{createInferBuilder(logger)};
    std::unique_ptr<INetworkDefinition, TrtDeleter<INetworkDefinition>> network{builder ? builder->createNetworkV2(0U) : nullptr};
    std::unique_ptr<IBuilderConfig, TrtDeleter<IBuilderConfig>> config{builder ? builder->createBuilderConfig() : nullptr};
    if (!builder || !network || !config)
    {
        std::fprintf(stderr, "create builder/network/config failed\n");
        return 1;
    }

    TensorFormats linear = 1U << static_cast<uint32_t>(TensorFormat::kLINEAR);
    ITensor* q = network->addInput("q_raw", DataType::kFLOAT, Dims4{kBatch, kSeqLen, kQHeads, kHeadDim});
    ITensor* k = network->addInput("k_raw", DataType::kFLOAT, Dims4{kBatch, kSeqLen, kKvHeads, kHeadDim});
    ITensor* v = network->addInput("v_raw", DataType::kFLOAT, Dims4{kBatch, kSeqLen, kKvHeads, kHeadDim});
    ITensor* pos = network->addInput("position_ids", DataType::kINT64, Dims2{kBatch, kSeqLen});
    ITensor* mask = network->addInput("attention_mask", DataType::kBOOL, Dims3{kBatch, kSeqLen, kSeqLen});
    for (ITensor* t : {q, k, v, pos, mask})
    {
        if (!t)
        {
            std::fprintf(stderr, "addInput failed\n");
            return 1;
        }
        t->setAllowedFormats(linear);
    }

    FusedAttentionRealV3 plugin{"fused_attention_v3_real"};
    ITensor* inputs[] = {q, k, v, pos, mask};
    IPluginV3Layer* layer = network->addPluginV3(inputs, 5, nullptr, 0, plugin);
    if (!layer)
    {
        std::fprintf(stderr, "addPluginV3 failed\n");
        return 1;
    }
    ITensor* out = layer->getOutput(0);
    out->setName("context");
    out->setAllowedFormats(linear);
    network->markOutput(*out);

    std::unique_ptr<IHostMemory, TrtDeleter<IHostMemory>> plan{builder->buildSerializedNetwork(*network, *config)};
    if (!plan || plan->size() == 0)
    {
        std::fprintf(stderr, "buildSerializedNetwork failed or returned empty plan\n");
        return 1;
    }
    std::filesystem::path planPath = outDir / "fused_attention_v3_real.plan";
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

    void *dq{}, *dk{}, *dv{}, *dp{}, *dm{}, *do_{};
    checkCuda(cudaMalloc(&dq, qCount * sizeof(float)), "cudaMalloc q");
    checkCuda(cudaMalloc(&dk, kvCount * sizeof(float)), "cudaMalloc k");
    checkCuda(cudaMalloc(&dv, kvCount * sizeof(float)), "cudaMalloc v");
    checkCuda(cudaMalloc(&dp, posCount * sizeof(int64_t)), "cudaMalloc pos");
    checkCuda(cudaMalloc(&dm, maskCount * sizeof(bool)), "cudaMalloc mask");
    checkCuda(cudaMalloc(&do_, outCount * sizeof(float)), "cudaMalloc out");
    checkCuda(cudaMemcpy(dq, hQ.data(), qCount * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy q");
    checkCuda(cudaMemcpy(dk, hK.data(), kvCount * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy k");
    checkCuda(cudaMemcpy(dv, hV.data(), kvCount * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy v");
    checkCuda(cudaMemcpy(dp, hPosBytes.data(), hPosBytes.size(), cudaMemcpyHostToDevice), "cudaMemcpy pos");
    checkCuda(cudaMemcpy(dm, hMaskBytes.data(), hMaskBytes.size(), cudaMemcpyHostToDevice), "cudaMemcpy mask");

    context->setTensorAddress("q_raw", dq);
    context->setTensorAddress("k_raw", dk);
    context->setTensorAddress("v_raw", dv);
    context->setTensorAddress("position_ids", dp);
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
    writeFloatFile(outDir / "context_plugin.fp32.bin", hOut);
    Metrics m = computeMetrics(hRef, hOut);
    double latencyMs = static_cast<double>(totalMs) / static_cast<double>(std::max(1, iters));

    std::filesystem::path reportPath = outDir / "fused_attention_v3_real_report.json";
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
           << "  \"plugin_latency_ms\": " << latencyMs << ",\n"
           << "  \"context_metrics\": {\n"
           << "    \"cosine\": " << m.cosine << ",\n"
           << "    \"relative_l2\": " << m.relativeL2 << ",\n"
           << "    \"max_abs\": " << m.maxAbs << "\n"
           << "  }\n"
           << "}\n";
    report.close();

    std::printf("built plan: %s (%zu bytes)\n", planPath.c_str(), plan->size());
    std::printf("enqueueV3 ok\n");
    std::printf("plugin latency: %.6f ms\n", latencyMs);
    std::printf("context cosine: %.9f relative_l2: %.9g max_abs: %.9g\n", m.cosine, m.relativeL2, m.maxAbs);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaStreamDestroy(stream);
    cudaFree(dq);
    cudaFree(dk);
    cudaFree(dv);
    cudaFree(dp);
    cudaFree(dm);
    cudaFree(do_);
    return 0;
}
