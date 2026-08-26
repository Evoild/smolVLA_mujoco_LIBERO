#pragma once

#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <NvInferRuntimePlugin.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <string>
#include <vector>

namespace smolvla
{

constexpr char const* kFusedRopeLayoutPluginName = "SmolVLAFusedRopeLayout";
constexpr char const* kFusedRopeLayoutPluginVersion = "1";
constexpr char const* kFusedRopeQKPluginName = "SmolVLAFusedRopeQK";
constexpr char const* kFusedRopeQKPluginVersion = "1";
constexpr char const* kFusedRopeQKSoftmaxPluginName = "SmolVLAFusedRopeQKSoftmax";
constexpr char const* kFusedRopeQKSoftmaxPluginVersion = "1";
constexpr char const* kFusedRopeAttentionPluginName = "SmolVLAFusedRopeAttention";
constexpr char const* kFusedRopeAttentionPluginVersion = "1";

struct FusedRopeLayoutParams
{
    int32_t qHeads{15};
    int32_t kvHeads{5};
    int32_t headDim{64};
    float maxWavelength{10000.0F};
};

int32_t launchFusedRopeLayout(void const* qRaw, void const* kRaw, void const* positionIds, void* qOut, void* kOut,
    int32_t batch, int32_t seqLen, FusedRopeLayoutParams const& params, nvinfer1::DataType dataType,
    nvinfer1::DataType positionType, cudaStream_t stream);

int32_t launchFusedRopeQK(void const* qRaw, void const* kRaw, void const* positionIds, void* scores,
    int32_t batch, int32_t seqLen, FusedRopeLayoutParams const& params, nvinfer1::DataType dataType,
    nvinfer1::DataType positionType, cudaStream_t stream);

int32_t launchFusedRopeQKSoftmax(void const* qRaw, void const* kRaw, void const* positionIds, void const* mask,
    void* probs, int32_t batch, int32_t seqLen, FusedRopeLayoutParams const& params, nvinfer1::DataType dataType,
    nvinfer1::DataType positionType, cudaStream_t stream);

int32_t launchFusedRopeAttention(void const* qRaw, void const* kRaw, void const* vRaw, void const* positionIds,
    void const* mask, void* context, int32_t batch, int32_t seqLen, FusedRopeLayoutParams const& params,
    nvinfer1::DataType dataType, nvinfer1::DataType positionType, cudaStream_t stream);

int32_t launchFusedRopeAttentionFp32MaskImplicitPos(void const* qRaw, void const* kRaw, void const* vRaw,
    void const* mask, void* context, int32_t batch, int32_t seqLen, FusedRopeLayoutParams const& params,
    cudaStream_t stream);

class FusedRopeLayoutPlugin final : public nvinfer1::IPluginV2DynamicExt
{
public:
    FusedRopeLayoutPlugin(char const* name, FusedRopeLayoutParams params);
    FusedRopeLayoutPlugin(char const* name, void const* serialData, size_t serialLength);

    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;
    nvinfer1::DimsExprs getOutputDimensions(
        int32_t outputIndex, nvinfer1::DimsExprs const* inputs, int32_t nbInputs, nvinfer1::IExprBuilder& exprBuilder)
        noexcept override;
    bool supportsFormatCombination(int32_t pos, nvinfer1::PluginTensorDesc const* inOut, int32_t nbInputs,
        int32_t nbOutputs) noexcept override;
    void configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int32_t nbInputs,
        nvinfer1::DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept override;
    size_t getWorkspaceSize(nvinfer1::PluginTensorDesc const* inputs, int32_t nbInputs,
        nvinfer1::PluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept override;
    int32_t enqueue(nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* outputDesc,
        void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;

    int32_t initialize() noexcept override;
    void terminate() noexcept override;
    size_t getSerializationSize() const noexcept override;
    void serialize(void* buffer) const noexcept override;
    void destroy() noexcept override;
    void setPluginNamespace(char const* pluginNamespace) noexcept override;
    char const* getPluginNamespace() const noexcept override;
    char const* getPluginType() const noexcept override;
    char const* getPluginVersion() const noexcept override;
    int32_t getNbOutputs() const noexcept override;
    nvinfer1::DataType getOutputDataType(int32_t index, nvinfer1::DataType const* inputTypes, int32_t nbInputs)
        const noexcept override;
    void attachToContext(
        cudnnContext* cudnnContext, cublasContext* cublasContext, nvinfer1::IGpuAllocator* gpuAllocator)
        noexcept override;
    void detachFromContext() noexcept override;

private:
    std::string mName;
    std::string mNamespace;
    FusedRopeLayoutParams mParams;
};

class FusedRopeLayoutPluginCreator final : public nvinfer1::IPluginCreator
{
public:
    FusedRopeLayoutPluginCreator();
    char const* getPluginName() const noexcept override;
    char const* getPluginVersion() const noexcept override;
    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override;
    nvinfer1::IPluginV2* createPlugin(char const* name, nvinfer1::PluginFieldCollection const* fc) noexcept override;
    nvinfer1::IPluginV2* deserializePlugin(char const* name, void const* serialData, size_t serialLength)
        noexcept override;
    void setPluginNamespace(char const* pluginNamespace) noexcept override;
    char const* getPluginNamespace() const noexcept override;

private:
    std::string mNamespace;
    std::vector<nvinfer1::PluginField> mFields;
    nvinfer1::PluginFieldCollection mFieldCollection{};
};

class FusedRopeQKPlugin final : public nvinfer1::IPluginV2DynamicExt
{
public:
    FusedRopeQKPlugin(char const* name, FusedRopeLayoutParams params);
    FusedRopeQKPlugin(char const* name, void const* serialData, size_t serialLength);

    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;
    nvinfer1::DimsExprs getOutputDimensions(
        int32_t outputIndex, nvinfer1::DimsExprs const* inputs, int32_t nbInputs, nvinfer1::IExprBuilder& exprBuilder)
        noexcept override;
    bool supportsFormatCombination(int32_t pos, nvinfer1::PluginTensorDesc const* inOut, int32_t nbInputs,
        int32_t nbOutputs) noexcept override;
    void configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int32_t nbInputs,
        nvinfer1::DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept override;
    size_t getWorkspaceSize(nvinfer1::PluginTensorDesc const* inputs, int32_t nbInputs,
        nvinfer1::PluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept override;
    int32_t enqueue(nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* outputDesc,
        void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;

    int32_t initialize() noexcept override;
    void terminate() noexcept override;
    size_t getSerializationSize() const noexcept override;
    void serialize(void* buffer) const noexcept override;
    void destroy() noexcept override;
    void setPluginNamespace(char const* pluginNamespace) noexcept override;
    char const* getPluginNamespace() const noexcept override;
    char const* getPluginType() const noexcept override;
    char const* getPluginVersion() const noexcept override;
    int32_t getNbOutputs() const noexcept override;
    nvinfer1::DataType getOutputDataType(int32_t index, nvinfer1::DataType const* inputTypes, int32_t nbInputs)
        const noexcept override;
    void attachToContext(
        cudnnContext* cudnnContext, cublasContext* cublasContext, nvinfer1::IGpuAllocator* gpuAllocator)
        noexcept override;
    void detachFromContext() noexcept override;

private:
    std::string mName;
    std::string mNamespace;
    FusedRopeLayoutParams mParams;
};

class FusedRopeQKPluginCreator final : public nvinfer1::IPluginCreator
{
public:
    FusedRopeQKPluginCreator();
    char const* getPluginName() const noexcept override;
    char const* getPluginVersion() const noexcept override;
    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override;
    nvinfer1::IPluginV2* createPlugin(char const* name, nvinfer1::PluginFieldCollection const* fc) noexcept override;
    nvinfer1::IPluginV2* deserializePlugin(char const* name, void const* serialData, size_t serialLength)
        noexcept override;
    void setPluginNamespace(char const* pluginNamespace) noexcept override;
    char const* getPluginNamespace() const noexcept override;

private:
    std::string mNamespace;
    std::vector<nvinfer1::PluginField> mFields;
    nvinfer1::PluginFieldCollection mFieldCollection{};
};

class FusedRopeQKSoftmaxPlugin final : public nvinfer1::IPluginV2DynamicExt
{
public:
    FusedRopeQKSoftmaxPlugin(char const* name, FusedRopeLayoutParams params);
    FusedRopeQKSoftmaxPlugin(char const* name, void const* serialData, size_t serialLength);

    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;
    nvinfer1::DimsExprs getOutputDimensions(
        int32_t outputIndex, nvinfer1::DimsExprs const* inputs, int32_t nbInputs, nvinfer1::IExprBuilder& exprBuilder)
        noexcept override;
    bool supportsFormatCombination(int32_t pos, nvinfer1::PluginTensorDesc const* inOut, int32_t nbInputs,
        int32_t nbOutputs) noexcept override;
    void configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int32_t nbInputs,
        nvinfer1::DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept override;
    size_t getWorkspaceSize(nvinfer1::PluginTensorDesc const* inputs, int32_t nbInputs,
        nvinfer1::PluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept override;
    int32_t enqueue(nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* outputDesc,
        void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;

    int32_t initialize() noexcept override;
    void terminate() noexcept override;
    size_t getSerializationSize() const noexcept override;
    void serialize(void* buffer) const noexcept override;
    void destroy() noexcept override;
    void setPluginNamespace(char const* pluginNamespace) noexcept override;
    char const* getPluginNamespace() const noexcept override;
    char const* getPluginType() const noexcept override;
    char const* getPluginVersion() const noexcept override;
    int32_t getNbOutputs() const noexcept override;
    nvinfer1::DataType getOutputDataType(int32_t index, nvinfer1::DataType const* inputTypes, int32_t nbInputs)
        const noexcept override;
    void attachToContext(
        cudnnContext* cudnnContext, cublasContext* cublasContext, nvinfer1::IGpuAllocator* gpuAllocator)
        noexcept override;
    void detachFromContext() noexcept override;

private:
    std::string mName;
    std::string mNamespace;
    FusedRopeLayoutParams mParams;
};

class FusedRopeQKSoftmaxPluginCreator final : public nvinfer1::IPluginCreator
{
public:
    FusedRopeQKSoftmaxPluginCreator();
    char const* getPluginName() const noexcept override;
    char const* getPluginVersion() const noexcept override;
    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override;
    nvinfer1::IPluginV2* createPlugin(char const* name, nvinfer1::PluginFieldCollection const* fc) noexcept override;
    nvinfer1::IPluginV2* deserializePlugin(char const* name, void const* serialData, size_t serialLength)
        noexcept override;
    void setPluginNamespace(char const* pluginNamespace) noexcept override;
    char const* getPluginNamespace() const noexcept override;

private:
    std::string mNamespace;
    std::vector<nvinfer1::PluginField> mFields;
    nvinfer1::PluginFieldCollection mFieldCollection{};
};

class FusedRopeAttentionPlugin final : public nvinfer1::IPluginV2DynamicExt
{
public:
    FusedRopeAttentionPlugin(char const* name, FusedRopeLayoutParams params);
    FusedRopeAttentionPlugin(char const* name, void const* serialData, size_t serialLength);

    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;
    nvinfer1::DimsExprs getOutputDimensions(
        int32_t outputIndex, nvinfer1::DimsExprs const* inputs, int32_t nbInputs, nvinfer1::IExprBuilder& exprBuilder)
        noexcept override;
    bool supportsFormatCombination(int32_t pos, nvinfer1::PluginTensorDesc const* inOut, int32_t nbInputs,
        int32_t nbOutputs) noexcept override;
    void configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int32_t nbInputs,
        nvinfer1::DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept override;
    size_t getWorkspaceSize(nvinfer1::PluginTensorDesc const* inputs, int32_t nbInputs,
        nvinfer1::PluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept override;
    int32_t enqueue(nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* outputDesc,
        void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;

    int32_t initialize() noexcept override;
    void terminate() noexcept override;
    size_t getSerializationSize() const noexcept override;
    void serialize(void* buffer) const noexcept override;
    void destroy() noexcept override;
    void setPluginNamespace(char const* pluginNamespace) noexcept override;
    char const* getPluginNamespace() const noexcept override;
    char const* getPluginType() const noexcept override;
    char const* getPluginVersion() const noexcept override;
    int32_t getNbOutputs() const noexcept override;
    nvinfer1::DataType getOutputDataType(int32_t index, nvinfer1::DataType const* inputTypes, int32_t nbInputs)
        const noexcept override;
    void attachToContext(
        cudnnContext* cudnnContext, cublasContext* cublasContext, nvinfer1::IGpuAllocator* gpuAllocator)
        noexcept override;
    void detachFromContext() noexcept override;

private:
    std::string mName;
    std::string mNamespace;
    FusedRopeLayoutParams mParams;
};

class FusedRopeAttentionPluginCreator final : public nvinfer1::IPluginCreator
{
public:
    FusedRopeAttentionPluginCreator();
    char const* getPluginName() const noexcept override;
    char const* getPluginVersion() const noexcept override;
    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override;
    nvinfer1::IPluginV2* createPlugin(char const* name, nvinfer1::PluginFieldCollection const* fc) noexcept override;
    nvinfer1::IPluginV2* deserializePlugin(char const* name, void const* serialData, size_t serialLength)
        noexcept override;
    void setPluginNamespace(char const* pluginNamespace) noexcept override;
    char const* getPluginNamespace() const noexcept override;

private:
    std::string mNamespace;
    std::vector<nvinfer1::PluginField> mFields;
    nvinfer1::PluginFieldCollection mFieldCollection{};
};

} // namespace smolvla
