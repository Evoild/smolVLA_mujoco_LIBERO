#include <NvInfer.h>
#include <NvInferPlugin.h>
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
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace nvinfer1;

namespace qkv_plugin_smoke
{

constexpr int32_t kBatch = 1;
constexpr int32_t kSeqLen = 177;
constexpr int32_t kM = 177;
constexpr int32_t kK = 960;
constexpr int32_t kNQ = 960;
constexpr int32_t kNKV = 320;
constexpr int32_t kQHeads = 15;
constexpr int32_t kKvHeads = 5;
constexpr int32_t kHeadDim = 64;
constexpr size_t kWorkspaceBytes = 16 * 1024 * 1024;
constexpr uintptr_t kWorkspaceAlignment = 256;

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

void checkCublas(cublasStatus_t status, char const* what)
{
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        std::fprintf(stderr, "%s failed: cublas status %d\n", what, static_cast<int>(status));
        std::exit(1);
    }
}

char* alignPtr(char* ptr)
{
    uintptr_t value = reinterpret_cast<uintptr_t>(ptr);
    value = (value + kWorkspaceAlignment - 1) & ~(kWorkspaceAlignment - 1);
    return reinterpret_cast<char*>(value);
}

Dims dims1(int32_t d0)
{
    Dims dims{};
    dims.nbDims = 1;
    dims.d[0] = d0;
    return dims;
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

__global__ void floatWeightToInt8Kernel(float const* src, int8_t* dst, int32_t total)
{
    int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total)
    {
        return;
    }
    dst[idx] = quantizeInt8(src[idx]);
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
};

void destroyPlan(LtPlan& p)
{
    if (p.pref) cublasLtMatmulPreferenceDestroy(p.pref);
    if (p.d) cublasLtMatrixLayoutDestroy(p.d);
    if (p.c) cublasLtMatrixLayoutDestroy(p.c);
    if (p.b) cublasLtMatrixLayoutDestroy(p.b);
    if (p.a) cublasLtMatrixLayoutDestroy(p.a);
    if (p.op) cublasLtMatmulDescDestroy(p.op);
    p = {};
}

LtPlan makePlan(cublasLtHandle_t handle, int32_t n)
{
    LtPlan p;
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
    checkCublas(cublasLtMatmulPreferenceSetAttribute(p.pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &kWorkspaceBytes,
                    sizeof(kWorkspaceBytes)),
        "workspace");
    int32_t returned = 0;
    checkCublas(cublasLtMatmulAlgoGetHeuristic(handle, p.op, p.a, p.b, p.c, p.d, p.pref, 1, &p.heuristic, &returned),
        "heuristic");
    if (returned <= 0 || p.heuristic.state != CUBLAS_STATUS_SUCCESS)
    {
        std::fprintf(stderr, "no cublasLt algorithm for n=%d\n", n);
        std::exit(1);
    }
    return p;
}

class SmoothQuantQkvPlugin final
    : public IPluginV3
    , public IPluginV3OneCore
    , public IPluginV3OneBuild
    , public IPluginV3OneRuntime
{
public:
    SmoothQuantQkvPlugin(std::string name, float qAs, float kAs, float vAs)
        : mName(std::move(name))
        , mQAs(qAs)
        , mKAs(kAs)
        , mVAs(vAs)
    {
        mSerializeFields.nbFields = 0;
        mSerializeFields.fields = nullptr;
    }

    ~SmoothQuantQkvPlugin() override { destroyLt(); }

    IPluginCapability* getCapabilityInterface(PluginCapabilityType type) noexcept override
    {
        if (type == PluginCapabilityType::kCORE) return static_cast<IPluginV3OneCore*>(this);
        if (type == PluginCapabilityType::kBUILD) return static_cast<IPluginV3OneBuild*>(this);
        if (type == PluginCapabilityType::kRUNTIME) return static_cast<IPluginV3OneRuntime*>(this);
        return nullptr;
    }

    IPluginV3* clone() noexcept override { return new SmoothQuantQkvPlugin(mName, mQAs, mKAs, mVAs); }
    AsciiChar const* getPluginName() const noexcept override { return "SmolVLASmoothQuantQKVV3Smoke"; }
    AsciiChar const* getPluginVersion() const noexcept override { return "1"; }
    AsciiChar const* getPluginNamespace() const noexcept override { return ""; }
    int32_t getNbOutputs() const noexcept override { return 3; }
    int32_t configurePlugin(DynamicPluginTensorDesc const*, int32_t, DynamicPluginTensorDesc const*, int32_t) noexcept override { return 0; }
    int32_t getFormatCombinationLimit() noexcept override { return 1; }
    int32_t onShapeChange(PluginTensorDesc const*, int32_t, PluginTensorDesc const*, int32_t) noexcept override { return 0; }
    PluginFieldCollection const* getFieldsToSerialize() noexcept override { return &mSerializeFields; }
    IPluginV3* attachToContext(IPluginResourceContext*) noexcept override { return clone(); }

    int32_t getOutputDataTypes(DataType* outputTypes, int32_t nbOutputs, DataType const*, int32_t) const noexcept override
    {
        if (nbOutputs != 3) return 1;
        outputTypes[0] = DataType::kFLOAT;
        outputTypes[1] = DataType::kFLOAT;
        outputTypes[2] = DataType::kFLOAT;
        return 0;
    }

    int32_t getOutputShapes(DimsExprs const* inputs, int32_t nbInputs, DimsExprs const*, int32_t, DimsExprs* outputs,
        int32_t nbOutputs, IExprBuilder& exprBuilder) noexcept override
    {
        if (nbInputs != 13 || nbOutputs != 3) return 1;
        outputs[0].nbDims = 4;
        outputs[0].d[0] = inputs[0].d[0];
        outputs[0].d[1] = inputs[0].d[1];
        outputs[0].d[2] = exprBuilder.constant(kQHeads);
        outputs[0].d[3] = exprBuilder.constant(kHeadDim);
        outputs[1].nbDims = 4;
        outputs[1].d[0] = inputs[0].d[0];
        outputs[1].d[1] = inputs[0].d[1];
        outputs[1].d[2] = exprBuilder.constant(kKvHeads);
        outputs[1].d[3] = exprBuilder.constant(kHeadDim);
        outputs[2] = outputs[1];
        return 0;
    }

    bool supportsFormatCombination(int32_t pos, DynamicPluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept override
    {
        if (nbInputs != 13 || nbOutputs != 3 || pos < 0 || pos >= 16) return false;
        if (inOut[pos].desc.format != TensorFormat::kLINEAR) return false;
        return inOut[pos].desc.type == DataType::kFLOAT;
    }

    size_t getWorkspaceSize(DynamicPluginTensorDesc const*, int32_t, DynamicPluginTensorDesc const*, int32_t) const noexcept override
    {
        size_t qx = static_cast<size_t>(kM) * kK;
        size_t kx = static_cast<size_t>(kM) * kK;
        size_t vx = static_cast<size_t>(kM) * kK;
        size_t qW = static_cast<size_t>(kNQ) * kK;
        size_t kW = static_cast<size_t>(kNKV) * kK;
        size_t vW = static_cast<size_t>(kNKV) * kK;
        size_t qAcc = static_cast<size_t>(kM) * kNQ * sizeof(int32_t);
        size_t kAcc = static_cast<size_t>(kM) * kNKV * sizeof(int32_t);
        size_t vAcc = static_cast<size_t>(kM) * kNKV * sizeof(int32_t);
        return qx + kx + vx + qW + kW + vW + qAcc + kAcc + vAcc + kWorkspaceBytes + 4096;
    }

    int32_t enqueue(PluginTensorDesc const*, PluginTensorDesc const*, void const* const* inputs, void* const* outputs,
        void* workspace, cudaStream_t stream) noexcept override
    {
        try
        {
            ensureLt();
            auto* cursor = static_cast<char*>(workspace);
            cursor = alignPtr(cursor);
            int8_t* qx = reinterpret_cast<int8_t*>(cursor);
            cursor += static_cast<size_t>(kM) * kK;
            cursor = alignPtr(cursor);
            int8_t* kx = reinterpret_cast<int8_t*>(cursor);
            cursor += static_cast<size_t>(kM) * kK;
            cursor = alignPtr(cursor);
            int8_t* vx = reinterpret_cast<int8_t*>(cursor);
            cursor += static_cast<size_t>(kM) * kK;
            cursor = alignPtr(cursor);
            int8_t* qW = reinterpret_cast<int8_t*>(cursor);
            cursor += static_cast<size_t>(kNQ) * kK;
            cursor = alignPtr(cursor);
            int8_t* kW = reinterpret_cast<int8_t*>(cursor);
            cursor += static_cast<size_t>(kNKV) * kK;
            cursor = alignPtr(cursor);
            int8_t* vW = reinterpret_cast<int8_t*>(cursor);
            cursor += static_cast<size_t>(kNKV) * kK;
            cursor = alignPtr(cursor);
            int32_t* qAcc = reinterpret_cast<int32_t*>(cursor);
            cursor += static_cast<size_t>(kM) * kNQ * sizeof(int32_t);
            cursor = alignPtr(cursor);
            int32_t* kAcc = reinterpret_cast<int32_t*>(cursor);
            cursor += static_cast<size_t>(kM) * kNKV * sizeof(int32_t);
            cursor = alignPtr(cursor);
            int32_t* vAcc = reinterpret_cast<int32_t*>(cursor);
            cursor += static_cast<size_t>(kM) * kNKV * sizeof(int32_t);
            cursor = alignPtr(cursor);
            void* ltWorkspace = cursor;

            auto const* x = static_cast<float const*>(inputs[0]);
            quantizeKernel<<<(kM * kK + 255) / 256, 256, 0, stream>>>(x, static_cast<float const*>(inputs[1]), qx, mQAs);
            quantizeKernel<<<(kM * kK + 255) / 256, 256, 0, stream>>>(x, static_cast<float const*>(inputs[2]), kx, mKAs);
            quantizeKernel<<<(kM * kK + 255) / 256, 256, 0, stream>>>(x, static_cast<float const*>(inputs[3]), vx, mVAs);
            floatWeightToInt8Kernel<<<(kNQ * kK + 255) / 256, 256, 0, stream>>>(static_cast<float const*>(inputs[4]), qW, kNQ * kK);
            floatWeightToInt8Kernel<<<(kNKV * kK + 255) / 256, 256, 0, stream>>>(static_cast<float const*>(inputs[5]), kW, kNKV * kK);
            floatWeightToInt8Kernel<<<(kNKV * kK + 255) / 256, 256, 0, stream>>>(static_cast<float const*>(inputs[6]), vW, kNKV * kK);
            matmul(mQPlan, qx, qW, qAcc, ltWorkspace, stream, "q");
            matmul(mKvPlan, kx, kW, kAcc, ltWorkspace, stream, "k");
            matmul(mKvPlan, vx, vW, vAcc, ltWorkspace, stream, "v");
            int32_t total = kM * (kNQ + 2 * kNKV);
            dequantQkvFusedKernel<<<(total + 255) / 256, 256, 0, stream>>>(qAcc, kAcc, vAcc,
                static_cast<float const*>(inputs[7]), static_cast<float const*>(inputs[8]), static_cast<float const*>(inputs[9]),
                static_cast<float const*>(inputs[10]), static_cast<float const*>(inputs[11]), static_cast<float const*>(inputs[12]),
                static_cast<float*>(outputs[0]), static_cast<float*>(outputs[1]), static_cast<float*>(outputs[2]), mQAs, mKAs, mVAs);
            return 0;
        }
        catch (...)
        {
            return 1;
        }
    }

private:
    void ensureLt()
    {
        if (mInitialized) return;
        checkCublas(cublasLtCreate(&mHandle), "plugin cublasLtCreate");
        mQPlan = makePlan(mHandle, kNQ);
        mKvPlan = makePlan(mHandle, kNKV);
        mInitialized = true;
    }

    void destroyLt()
    {
        destroyPlan(mQPlan);
        destroyPlan(mKvPlan);
        if (mHandle)
        {
            cublasLtDestroy(mHandle);
            mHandle = nullptr;
        }
        mInitialized = false;
    }

    void matmul(LtPlan& p, int8_t const* a, int8_t const* b, int32_t* d, void* workspace, cudaStream_t stream, char const* tag)
    {
        int32_t alpha = 1;
        int32_t beta = 0;
        cublasStatus_t status = cublasLtMatmul(mHandle, p.op, &alpha, a, p.a, b, p.b, &beta, d, p.c, d, p.d,
            &p.heuristic.algo, workspace, kWorkspaceBytes, stream);
        if (status != CUBLAS_STATUS_SUCCESS)
        {
            std::fprintf(stderr, "plugin cublasLtMatmul[%s] failed: cublas status %d a=%p b=%p d=%p workspace=%p\n",
                tag, static_cast<int>(status), static_cast<void const*>(a), static_cast<void const*>(b),
                static_cast<void*>(d), workspace);
            throw std::runtime_error("cublasLtMatmul failed");
        }
    }

    std::string mName;
    float mQAs{};
    float mKAs{};
    float mVAs{};
    PluginFieldCollection mSerializeFields{};
    cublasLtHandle_t mHandle{};
    LtPlan mQPlan{};
    LtPlan mKvPlan{};
    bool mInitialized{false};
};

class SmoothQuantQkvPluginCreator final : public IPluginCreatorV3One
{
public:
    SmoothQuantQkvPluginCreator()
    {
        mFields.nbFields = 0;
        mFields.fields = nullptr;
    }

    IPluginV3* createPlugin(AsciiChar const* name, PluginFieldCollection const*, TensorRTPhase) noexcept override
    {
        return new SmoothQuantQkvPlugin(name ? name : "smoothquant_qkv", mQAs, mKAs, mVAs);
    }

    PluginFieldCollection const* getFieldNames() noexcept override { return &mFields; }
    AsciiChar const* getPluginName() const noexcept override { return "SmolVLASmoothQuantQKVV3Smoke"; }
    AsciiChar const* getPluginVersion() const noexcept override { return "1"; }
    AsciiChar const* getPluginNamespace() const noexcept override { return ""; }

    void setScales(float qAs, float kAs, float vAs)
    {
        mQAs = qAs;
        mKAs = kAs;
        mVAs = vAs;
    }

private:
    PluginFieldCollection mFields{};
    float mQAs{};
    float mKAs{};
    float mVAs{};
};

void mallocCopy(void** dst, void const* src, size_t bytes, char const* name)
{
    checkCuda(cudaMalloc(dst, bytes), name);
    checkCuda(cudaMemcpy(*dst, src, bytes, cudaMemcpyHostToDevice), name);
}

float timeCuda(cudaStream_t stream, int32_t warmup, int32_t iters, IExecutionContext& ctx)
{
    for (int32_t i = 0; i < warmup; ++i)
    {
        if (!ctx.enqueueV3(stream))
        {
            std::fprintf(stderr, "enqueueV3 failed during warmup\n");
            std::exit(1);
        }
    }
    checkCuda(cudaStreamSynchronize(stream), "warmup sync");
    cudaEvent_t start{}, stop{};
    checkCuda(cudaEventCreate(&start), "event start");
    checkCuda(cudaEventCreate(&stop), "event stop");
    checkCuda(cudaEventRecord(start, stream), "record start");
    for (int32_t i = 0; i < iters; ++i)
    {
        if (!ctx.enqueueV3(stream))
        {
            std::fprintf(stderr, "enqueueV3 failed during timing\n");
            std::exit(1);
        }
    }
    checkCuda(cudaEventRecord(stop, stream), "record stop");
    checkCuda(cudaEventSynchronize(stop), "sync stop");
    float total = 0.0F;
    checkCuda(cudaEventElapsedTime(&total, start, stop), "elapsed");
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return total / static_cast<float>(std::max(1, iters));
}

} // namespace qkv_plugin_smoke

using namespace qkv_plugin_smoke;

int main(int argc, char** argv)
{
    std::filesystem::path dataDir = argc > 1 ? argv[1] : "runs/deploy/6/H-qkv-smoothquant-cublaslt/tensors";
    std::filesystem::path outDir = argc > 2 ? argv[2] : "runs/deploy/6/I-qkv-plugin-network-api";
    float qAs = argc > 3 ? std::atof(argv[3]) : 1.0F;
    float kAs = argc > 4 ? std::atof(argv[4]) : 1.0F;
    float vAs = argc > 5 ? std::atof(argv[5]) : 1.0F;
    int32_t warmup = argc > 6 ? std::atoi(argv[6]) : 20;
    int32_t iters = argc > 7 ? std::atoi(argv[7]) : 200;
    std::filesystem::create_directories(outDir);

    Logger logger;
    SmoothQuantQkvPluginCreator creator;
    creator.setScales(qAs, kAs, vAs);
    getPluginRegistry()->registerCreator(creator, "");
    std::unique_ptr<IBuilder, TrtDeleter<IBuilder>> builder{createInferBuilder(logger)};
    std::unique_ptr<INetworkDefinition, TrtDeleter<INetworkDefinition>> network{builder ? builder->createNetworkV2(0U) : nullptr};
    std::unique_ptr<IBuilderConfig, TrtDeleter<IBuilderConfig>> config{builder ? builder->createBuilderConfig() : nullptr};
    if (!builder || !network || !config)
    {
        std::fprintf(stderr, "create builder/network/config failed\n");
        return 1;
    }

    TensorFormats linear = 1U << static_cast<uint32_t>(TensorFormat::kLINEAR);
    ITensor* inputs[13]{};
    inputs[0] = network->addInput("x", DataType::kFLOAT, Dims3{kBatch, kSeqLen, kK});
    inputs[1] = network->addInput("q_smooth", DataType::kFLOAT, dims1(kK));
    inputs[2] = network->addInput("k_smooth", DataType::kFLOAT, dims1(kK));
    inputs[3] = network->addInput("v_smooth", DataType::kFLOAT, dims1(kK));
    inputs[4] = network->addInput("q_weight", DataType::kFLOAT, Dims2{kNQ, kK});
    inputs[5] = network->addInput("k_weight", DataType::kFLOAT, Dims2{kNKV, kK});
    inputs[6] = network->addInput("v_weight", DataType::kFLOAT, Dims2{kNKV, kK});
    inputs[7] = network->addInput("q_weight_scale", DataType::kFLOAT, dims1(kNQ));
    inputs[8] = network->addInput("k_weight_scale", DataType::kFLOAT, dims1(kNKV));
    inputs[9] = network->addInput("v_weight_scale", DataType::kFLOAT, dims1(kNKV));
    inputs[10] = network->addInput("q_bias", DataType::kFLOAT, dims1(kNQ));
    inputs[11] = network->addInput("k_bias", DataType::kFLOAT, dims1(kNKV));
    inputs[12] = network->addInput("v_bias", DataType::kFLOAT, dims1(kNKV));
    for (ITensor* t : inputs)
    {
        if (!t)
        {
            std::fprintf(stderr, "addInput failed\n");
            return 1;
        }
        t->setAllowedFormats(linear);
    }
    SmoothQuantQkvPlugin plugin{"smoothquant_qkv", qAs, kAs, vAs};
    IPluginV3Layer* layer = network->addPluginV3(inputs, 13, nullptr, 0, plugin);
    if (!layer)
    {
        std::fprintf(stderr, "addPluginV3 failed\n");
        return 1;
    }
    char const* outputNames[] = {"q_raw", "k_raw", "v_raw"};
    for (int32_t i = 0; i < 3; ++i)
    {
        ITensor* out = layer->getOutput(i);
        out->setName(outputNames[i]);
        out->setAllowedFormats(linear);
        network->markOutput(*out);
    }

    std::unique_ptr<IHostMemory, TrtDeleter<IHostMemory>> plan{builder->buildSerializedNetwork(*network, *config)};
    if (!plan || plan->size() == 0)
    {
        std::fprintf(stderr, "buildSerializedNetwork failed or returned empty plan\n");
        return 1;
    }
    std::filesystem::path planPath = outDir / "smoothquant_qkv_plugin_v3_smoke.plan";
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

    auto hX = readFloatFile(dataDir / "x.fp32.bin", static_cast<size_t>(kM) * kK);
    auto hQS = readFloatFile(dataDir / "q_smooth_scale.fp32.bin", kK);
    auto hKS = readFloatFile(dataDir / "k_smooth_scale.fp32.bin", kK);
    auto hVS = readFloatFile(dataDir / "v_smooth_scale.fp32.bin", kK);
    auto hQw = readInt8File(dataDir / "q_qweight.int8.bin", static_cast<size_t>(kNQ) * kK);
    auto hKw = readInt8File(dataDir / "k_qweight.int8.bin", static_cast<size_t>(kNKV) * kK);
    auto hVw = readInt8File(dataDir / "v_qweight.int8.bin", static_cast<size_t>(kNKV) * kK);
    std::vector<float> hQwFloat(hQw.size());
    std::vector<float> hKwFloat(hKw.size());
    std::vector<float> hVwFloat(hVw.size());
    std::transform(hQw.begin(), hQw.end(), hQwFloat.begin(), [](int8_t value) { return static_cast<float>(value); });
    std::transform(hKw.begin(), hKw.end(), hKwFloat.begin(), [](int8_t value) { return static_cast<float>(value); });
    std::transform(hVw.begin(), hVw.end(), hVwFloat.begin(), [](int8_t value) { return static_cast<float>(value); });
    auto hQWs = readFloatFile(dataDir / "q_weight_scale.fp32.bin", kNQ);
    auto hKWs = readFloatFile(dataDir / "k_weight_scale.fp32.bin", kNKV);
    auto hVWs = readFloatFile(dataDir / "v_weight_scale.fp32.bin", kNKV);
    auto hQBias = readFloatFile(dataDir / "q_bias.fp32.bin", kNQ);
    auto hKBias = readFloatFile(dataDir / "k_bias.fp32.bin", kNKV);
    auto hVBias = readFloatFile(dataDir / "v_bias.fp32.bin", kNKV);
    auto hQRef = readFloatFile(dataDir / "q_reference.fp32.bin", static_cast<size_t>(kM) * kNQ);
    auto hKRef = readFloatFile(dataDir / "k_reference.fp32.bin", static_cast<size_t>(kM) * kNKV);
    auto hVRef = readFloatFile(dataDir / "v_reference.fp32.bin", static_cast<size_t>(kM) * kNKV);

    void* dev[16]{};
    mallocCopy(&dev[0], hX.data(), hX.size() * sizeof(float), "x");
    mallocCopy(&dev[1], hQS.data(), hQS.size() * sizeof(float), "q smooth");
    mallocCopy(&dev[2], hKS.data(), hKS.size() * sizeof(float), "k smooth");
    mallocCopy(&dev[3], hVS.data(), hVS.size() * sizeof(float), "v smooth");
    mallocCopy(&dev[4], hQwFloat.data(), hQwFloat.size() * sizeof(float), "q weight");
    mallocCopy(&dev[5], hKwFloat.data(), hKwFloat.size() * sizeof(float), "k weight");
    mallocCopy(&dev[6], hVwFloat.data(), hVwFloat.size() * sizeof(float), "v weight");
    mallocCopy(&dev[7], hQWs.data(), hQWs.size() * sizeof(float), "q ws");
    mallocCopy(&dev[8], hKWs.data(), hKWs.size() * sizeof(float), "k ws");
    mallocCopy(&dev[9], hVWs.data(), hVWs.size() * sizeof(float), "v ws");
    mallocCopy(&dev[10], hQBias.data(), hQBias.size() * sizeof(float), "q bias");
    mallocCopy(&dev[11], hKBias.data(), hKBias.size() * sizeof(float), "k bias");
    mallocCopy(&dev[12], hVBias.data(), hVBias.size() * sizeof(float), "v bias");
    checkCuda(cudaMalloc(&dev[13], hQRef.size() * sizeof(float)), "q out");
    checkCuda(cudaMalloc(&dev[14], hKRef.size() * sizeof(float)), "k out");
    checkCuda(cudaMalloc(&dev[15], hVRef.size() * sizeof(float)), "v out");

    char const* names[] = {"x", "q_smooth", "k_smooth", "v_smooth", "q_weight", "k_weight", "v_weight",
        "q_weight_scale", "k_weight_scale", "v_weight_scale", "q_bias", "k_bias", "v_bias", "q_raw", "k_raw", "v_raw"};
    for (int32_t i = 0; i < 16; ++i)
    {
        context->setTensorAddress(names[i], dev[i]);
    }
    cudaStream_t stream{};
    checkCuda(cudaStreamCreate(&stream), "stream");
    float latencyMs = timeCuda(stream, warmup, iters, *context);

    std::vector<float> hQOut(hQRef.size());
    std::vector<float> hKOut(hKRef.size());
    std::vector<float> hVOut(hVRef.size());
    checkCuda(cudaMemcpy(hQOut.data(), dev[13], hQOut.size() * sizeof(float), cudaMemcpyDeviceToHost), "q copy");
    checkCuda(cudaMemcpy(hKOut.data(), dev[14], hKOut.size() * sizeof(float), cudaMemcpyDeviceToHost), "k copy");
    checkCuda(cudaMemcpy(hVOut.data(), dev[15], hVOut.size() * sizeof(float), cudaMemcpyDeviceToHost), "v copy");
    Metrics qm = computeMetrics(hQRef, hQOut);
    Metrics km = computeMetrics(hKRef, hKOut);
    Metrics vm = computeMetrics(hVRef, hVOut);

    std::ofstream report(outDir / "smoothquant_qkv_plugin_v3_smoke_report.json");
    report << "{\n"
           << "  \"plan\": \"" << planPath.string() << "\",\n"
           << "  \"plan_size_bytes\": " << plan->size() << ",\n"
           << "  \"warmup\": " << warmup << ", \"iters\": " << iters << ",\n"
           << "  \"plugin_latency_ms\": " << latencyMs << ",\n"
           << "  \"metrics\": {\n"
           << "    \"q\": {\"cosine\": " << qm.cosine << ", \"relative_l2\": " << qm.relativeL2 << ", \"max_abs\": " << qm.maxAbs << "},\n"
           << "    \"k\": {\"cosine\": " << km.cosine << ", \"relative_l2\": " << km.relativeL2 << ", \"max_abs\": " << km.maxAbs << "},\n"
           << "    \"v\": {\"cosine\": " << vm.cosine << ", \"relative_l2\": " << vm.relativeL2 << ", \"max_abs\": " << vm.maxAbs << "}\n"
           << "  }\n"
           << "}\n";
    report.close();
    std::printf("built plan: %s (%zu bytes)\n", planPath.c_str(), plan->size());
    std::printf("enqueueV3 ok\n");
    std::printf("plugin latency: %.6f ms\n", latencyMs);
    std::printf("q rel_l2 %.9g k rel_l2 %.9g v rel_l2 %.9g\n", qm.relativeL2, km.relativeL2, vm.relativeL2);

    cudaStreamDestroy(stream);
    for (void* ptr : dev)
    {
        cudaFree(ptr);
    }
    return 0;
}
