#include "fused_rope_layout.h"

#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

using namespace nvinfer1;

namespace
{

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
    void operator()(T* p) const
    {
        delete p;
    }
};

class FusedAttentionV3 final
    : public IPluginV3
    , public IPluginV3OneCore
    , public IPluginV3OneBuild
    , public IPluginV3OneRuntime
{
public:
    explicit FusedAttentionV3(std::string name)
        : mName(std::move(name))
    {
        mParams.qHeads = 15;
        mParams.kvHeads = 5;
        mParams.headDim = 64;
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

    IPluginV3* clone() noexcept override { return new FusedAttentionV3(mName); }

    AsciiChar const* getPluginName() const noexcept override { return "SmolVLAFusedRopeAttentionV3Smoke"; }
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
        if (nbInputs != 4 || nbOutputs != 1)
        {
            return 1;
        }
        outputs[0].nbDims = 4;
        outputs[0].d[0] = inputs[0].d[0];
        outputs[0].d[1] = exprBuilder.constant(mParams.qHeads);
        outputs[0].d[2] = inputs[0].d[1];
        outputs[0].d[3] = exprBuilder.constant(mParams.headDim);
        return 0;
    }

    bool supportsFormatCombination(
        int32_t pos, DynamicPluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept override
    {
        if (nbInputs != 4 || nbOutputs != 1 || pos < 0 || pos >= 5)
        {
            return false;
        }
        return inOut[pos].desc.format == TensorFormat::kLINEAR && inOut[pos].desc.type == DataType::kFLOAT;
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
        return smolvla::launchFusedRopeAttentionFp32MaskImplicitPos(
            inputs[0], inputs[1], inputs[2], inputs[3], outputs[0], batch, seqLen, mParams, stream);
    }

    IPluginV3* attachToContext(IPluginResourceContext*) noexcept override { return clone(); }

    PluginFieldCollection const* getFieldsToSerialize() noexcept override { return &mSerializeFields; }

private:
    std::string mName;
    smolvla::FusedRopeLayoutParams mParams{};
    PluginFieldCollection mSerializeFields{};
};

class FusedAttentionV3Creator final : public IPluginCreatorV3One
{
public:
    FusedAttentionV3Creator()
    {
        mFields.nbFields = 0;
        mFields.fields = nullptr;
    }

    IPluginV3* createPlugin(AsciiChar const* name, PluginFieldCollection const*, TensorRTPhase) noexcept override
    {
        return new FusedAttentionV3(name ? name : "fused_attention_v3_smoke");
    }

    PluginFieldCollection const* getFieldNames() noexcept override { return &mFields; }
    AsciiChar const* getPluginName() const noexcept override { return "SmolVLAFusedRopeAttentionV3Smoke"; }
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

} // namespace

int main(int argc, char** argv)
{
    std::string outPlan = argc > 1 ? argv[1] : "runs/deploy/6/E-network-api-layer0-attention/fused_attention_v3_smoke.plan";
    Logger logger;
    FusedAttentionV3Creator creator;
    getPluginRegistry()->registerCreator(creator, "");

    std::unique_ptr<IBuilder, TrtDeleter<IBuilder>> builder{createInferBuilder(logger)};
    if (!builder)
    {
        std::fprintf(stderr, "createInferBuilder failed\n");
        return 1;
    }
    std::unique_ptr<INetworkDefinition, TrtDeleter<INetworkDefinition>> network{builder->createNetworkV2(0U)};
    std::unique_ptr<IBuilderConfig, TrtDeleter<IBuilderConfig>> config{builder->createBuilderConfig()};
    if (!network || !config)
    {
        std::fprintf(stderr, "create network/config failed\n");
        return 1;
    }

    TensorFormats linear = 1U << static_cast<uint32_t>(TensorFormat::kLINEAR);
    ITensor* q = network->addInput("q_raw", DataType::kFLOAT, Dims3{1, 177, 960});
    ITensor* k = network->addInput("k_raw", DataType::kFLOAT, Dims3{1, 177, 320});
    ITensor* v = network->addInput("v_raw", DataType::kFLOAT, Dims3{1, 177, 320});
    ITensor* mask = network->addInput("mask", DataType::kFLOAT, Dims3{1, 177, 177});
    for (ITensor* t : {q, k, v, mask})
    {
        if (!t)
        {
            std::fprintf(stderr, "addInput failed\n");
            return 1;
        }
        t->setAllowedFormats(linear);
    }

    FusedAttentionV3 plugin{"fused_attention_v3_smoke"};
    ITensor* inputs[] = {q, k, v, mask};
    IPluginV3Layer* layer = network->addPluginV3(inputs, 4, nullptr, 0, plugin);
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
    std::ofstream os(outPlan, std::ios::binary);
    os.write(static_cast<char const*>(plan->data()), static_cast<std::streamsize>(plan->size()));
    os.close();
    std::printf("built plan: %s (%zu bytes)\n", outPlan.c_str(), plan->size());

    std::unique_ptr<IRuntime, TrtDeleter<IRuntime>> runtime{createInferRuntime(logger)};
    if (!runtime)
    {
        std::fprintf(stderr, "createInferRuntime failed\n");
        return 1;
    }
    std::unique_ptr<ICudaEngine, TrtDeleter<ICudaEngine>> engine{
        runtime->deserializeCudaEngine(plan->data(), plan->size())};
    if (!engine)
    {
        std::fprintf(stderr, "deserializeCudaEngine failed\n");
        return 1;
    }
    std::unique_ptr<IExecutionContext, TrtDeleter<IExecutionContext>> context{engine->createExecutionContext()};
    if (!context)
    {
        std::fprintf(stderr, "createExecutionContext failed\n");
        return 1;
    }

    size_t qBytes = 177U * 960U * sizeof(float);
    size_t kvBytes = 177U * 320U * sizeof(float);
    size_t maskBytes = 177U * 177U * sizeof(float);
    size_t outBytes = 15U * 177U * 64U * sizeof(float);
    void *dq{}, *dk{}, *dv{}, *dm{}, *do_{};
    checkCuda(cudaMalloc(&dq, qBytes), "cudaMalloc q");
    checkCuda(cudaMalloc(&dk, kvBytes), "cudaMalloc k");
    checkCuda(cudaMalloc(&dv, kvBytes), "cudaMalloc v");
    checkCuda(cudaMalloc(&dm, maskBytes), "cudaMalloc mask");
    checkCuda(cudaMalloc(&do_, outBytes), "cudaMalloc out");
    checkCuda(cudaMemset(dq, 0, qBytes), "cudaMemset q");
    checkCuda(cudaMemset(dk, 0, kvBytes), "cudaMemset k");
    checkCuda(cudaMemset(dv, 0, kvBytes), "cudaMemset v");
    std::vector<float> hMask(177U * 177U, 0.0F);
    for (int i = 0; i < 177; ++i)
    {
        for (int j = 0; j <= i; ++j)
        {
            hMask[static_cast<size_t>(i) * 177U + static_cast<size_t>(j)] = 1.0F;
        }
    }
    checkCuda(cudaMemcpy(dm, hMask.data(), maskBytes, cudaMemcpyHostToDevice), "cudaMemcpy mask");

    context->setTensorAddress("q_raw", dq);
    context->setTensorAddress("k_raw", dk);
    context->setTensorAddress("v_raw", dv);
    context->setTensorAddress("mask", dm);
    context->setTensorAddress("context", do_);
    cudaStream_t stream{};
    checkCuda(cudaStreamCreate(&stream), "cudaStreamCreate");
    if (!context->enqueueV3(stream))
    {
        std::fprintf(stderr, "enqueueV3 failed\n");
        return 1;
    }
    checkCuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
    std::printf("enqueueV3 ok\n");

    cudaStreamDestroy(stream);
    cudaFree(dq);
    cudaFree(dk);
    cudaFree(dv);
    cudaFree(dm);
    cudaFree(do_);
    return 0;
}
