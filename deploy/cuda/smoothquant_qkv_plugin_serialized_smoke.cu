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
#include <utility>
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
    SmoothQuantQkvPlugin(std::string name, float qAs, float kAs, float vAs, std::vector<float> qSmooth,
        std::vector<float> kSmooth, std::vector<float> vSmooth, std::vector<int8_t> qWeight,
        std::vector<int8_t> kWeight, std::vector<int8_t> vWeight, std::vector<float> qWeightScale,
        std::vector<float> kWeightScale, std::vector<float> vWeightScale, std::vector<float> qBias,
        std::vector<float> kBias, std::vector<float> vBias)
        : mName(std::move(name))
        , mQAs(qAs)
        , mKAs(kAs)
        , mVAs(vAs)
        , mQSmooth(std::move(qSmooth))
        , mKSmooth(std::move(kSmooth))
        , mVSmooth(std::move(vSmooth))
        , mQWeight(std::move(qWeight))
        , mKWeight(std::move(kWeight))
        , mVWeight(std::move(vWeight))
        , mQWeightScale(std::move(qWeightScale))
        , mKWeightScale(std::move(kWeightScale))
        , mVWeightScale(std::move(vWeightScale))
        , mQBias(std::move(qBias))
        , mKBias(std::move(kBias))
        , mVBias(std::move(vBias))
    {
        buildSerializeFields();
    }

    ~SmoothQuantQkvPlugin() override { destroyPersistent(); }

    IPluginCapability* getCapabilityInterface(PluginCapabilityType type) noexcept override
    {
        if (type == PluginCapabilityType::kCORE) return static_cast<IPluginV3OneCore*>(this);
        if (type == PluginCapabilityType::kBUILD) return static_cast<IPluginV3OneBuild*>(this);
        if (type == PluginCapabilityType::kRUNTIME) return static_cast<IPluginV3OneRuntime*>(this);
        return nullptr;
    }

    IPluginV3* clone() noexcept override
    {
        return new SmoothQuantQkvPlugin(mName, mQAs, mKAs, mVAs, mQSmooth, mKSmooth, mVSmooth, mQWeight, mKWeight,
            mVWeight, mQWeightScale, mKWeightScale, mVWeightScale, mQBias, mKBias, mVBias);
    }
    AsciiChar const* getPluginName() const noexcept override { return "SmolVLASmoothQuantQKVSerializedV3Smoke"; }
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
        if (nbInputs != 1 || nbOutputs != 3) return 1;
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
        if (nbInputs != 1 || nbOutputs != 3 || pos < 0 || pos >= 4) return false;
        if (inOut[pos].desc.format != TensorFormat::kLINEAR) return false;
        return inOut[pos].desc.type == DataType::kFLOAT;
    }

    size_t getWorkspaceSize(DynamicPluginTensorDesc const*, int32_t, DynamicPluginTensorDesc const*, int32_t) const noexcept override
    {
        size_t qx = static_cast<size_t>(kM) * kK;
        size_t kx = static_cast<size_t>(kM) * kK;
        size_t vx = static_cast<size_t>(kM) * kK;
        size_t qAcc = static_cast<size_t>(kM) * kNQ * sizeof(int32_t);
        size_t kAcc = static_cast<size_t>(kM) * kNKV * sizeof(int32_t);
        size_t vAcc = static_cast<size_t>(kM) * kNKV * sizeof(int32_t);
        return qx + kx + vx + qAcc + kAcc + vAcc + kWorkspaceBytes + 4096;
    }

    int32_t enqueue(PluginTensorDesc const*, PluginTensorDesc const*, void const* const* inputs, void* const* outputs,
        void* workspace, cudaStream_t stream) noexcept override
    {
        try
        {
            ensurePersistent();
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
            quantizeKernel<<<(kM * kK + 255) / 256, 256, 0, stream>>>(x, mDQSmooth, qx, mQAs);
            quantizeKernel<<<(kM * kK + 255) / 256, 256, 0, stream>>>(x, mDKSmooth, kx, mKAs);
            quantizeKernel<<<(kM * kK + 255) / 256, 256, 0, stream>>>(x, mDVSmooth, vx, mVAs);
            matmul(mQPlan, qx, mDQWeight, qAcc, ltWorkspace, stream, "q");
            matmul(mKvPlan, kx, mDKWeight, kAcc, ltWorkspace, stream, "k");
            matmul(mKvPlan, vx, mDVWeight, vAcc, ltWorkspace, stream, "v");
            int32_t total = kM * (kNQ + 2 * kNKV);
            dequantQkvFusedKernel<<<(total + 255) / 256, 256, 0, stream>>>(qAcc, kAcc, vAcc,
                mDQWeightScale, mDKWeightScale, mDVWeightScale, mDQBias, mDKBias, mDVBias,
                static_cast<float*>(outputs[0]), static_cast<float*>(outputs[1]), static_cast<float*>(outputs[2]), mQAs, mKAs, mVAs);
            return 0;
        }
        catch (...)
        {
            return 1;
        }
    }

private:
    template <typename T>
    void upload(T** dst, std::vector<T> const& src, char const* name)
    {
        if (*dst != nullptr)
        {
            return;
        }
        checkCuda(cudaMalloc(reinterpret_cast<void**>(dst), src.size() * sizeof(T)), name);
        checkCuda(cudaMemcpy(*dst, src.data(), src.size() * sizeof(T), cudaMemcpyHostToDevice), name);
    }

    void ensurePersistent()
    {
        if (mInitialized) return;
        upload(&mDQSmooth, mQSmooth, "upload q smooth");
        upload(&mDKSmooth, mKSmooth, "upload k smooth");
        upload(&mDVSmooth, mVSmooth, "upload v smooth");
        upload(&mDQWeight, mQWeight, "upload q weight");
        upload(&mDKWeight, mKWeight, "upload k weight");
        upload(&mDVWeight, mVWeight, "upload v weight");
        upload(&mDQWeightScale, mQWeightScale, "upload q weight scale");
        upload(&mDKWeightScale, mKWeightScale, "upload k weight scale");
        upload(&mDVWeightScale, mVWeightScale, "upload v weight scale");
        upload(&mDQBias, mQBias, "upload q bias");
        upload(&mDKBias, mKBias, "upload k bias");
        upload(&mDVBias, mVBias, "upload v bias");
        checkCublas(cublasLtCreate(&mHandle), "plugin cublasLtCreate");
        mQPlan = makePlan(mHandle, kNQ);
        mKvPlan = makePlan(mHandle, kNKV);
        mInitialized = true;
    }

    void destroyPersistent()
    {
        destroyPlan(mQPlan);
        destroyPlan(mKvPlan);
        if (mHandle)
        {
            cublasLtDestroy(mHandle);
            mHandle = nullptr;
        }
        cudaFree(mDQSmooth);
        cudaFree(mDKSmooth);
        cudaFree(mDVSmooth);
        cudaFree(mDQWeight);
        cudaFree(mDKWeight);
        cudaFree(mDVWeight);
        cudaFree(mDQWeightScale);
        cudaFree(mDKWeightScale);
        cudaFree(mDVWeightScale);
        cudaFree(mDQBias);
        cudaFree(mDKBias);
        cudaFree(mDVBias);
        mDQSmooth = nullptr;
        mDKSmooth = nullptr;
        mDVSmooth = nullptr;
        mDQWeight = nullptr;
        mDKWeight = nullptr;
        mDVWeight = nullptr;
        mDQWeightScale = nullptr;
        mDKWeightScale = nullptr;
        mDVWeightScale = nullptr;
        mDQBias = nullptr;
        mDKBias = nullptr;
        mDVBias = nullptr;
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

    void buildSerializeFields()
    {
        mSerializeFieldStorage.clear();
        mSerializeFieldStorage.emplace_back("q_activation_scale", &mQAs, PluginFieldType::kFLOAT32, 1);
        mSerializeFieldStorage.emplace_back("k_activation_scale", &mKAs, PluginFieldType::kFLOAT32, 1);
        mSerializeFieldStorage.emplace_back("v_activation_scale", &mVAs, PluginFieldType::kFLOAT32, 1);
        mSerializeFieldStorage.emplace_back("q_smooth_scale", mQSmooth.data(), PluginFieldType::kFLOAT32,
            static_cast<int32_t>(mQSmooth.size()));
        mSerializeFieldStorage.emplace_back("k_smooth_scale", mKSmooth.data(), PluginFieldType::kFLOAT32,
            static_cast<int32_t>(mKSmooth.size()));
        mSerializeFieldStorage.emplace_back("v_smooth_scale", mVSmooth.data(), PluginFieldType::kFLOAT32,
            static_cast<int32_t>(mVSmooth.size()));
        mSerializeFieldStorage.emplace_back("q_weight", mQWeight.data(), PluginFieldType::kINT8,
            static_cast<int32_t>(mQWeight.size()));
        mSerializeFieldStorage.emplace_back("k_weight", mKWeight.data(), PluginFieldType::kINT8,
            static_cast<int32_t>(mKWeight.size()));
        mSerializeFieldStorage.emplace_back("v_weight", mVWeight.data(), PluginFieldType::kINT8,
            static_cast<int32_t>(mVWeight.size()));
        mSerializeFieldStorage.emplace_back("q_weight_scale", mQWeightScale.data(), PluginFieldType::kFLOAT32,
            static_cast<int32_t>(mQWeightScale.size()));
        mSerializeFieldStorage.emplace_back("k_weight_scale", mKWeightScale.data(), PluginFieldType::kFLOAT32,
            static_cast<int32_t>(mKWeightScale.size()));
        mSerializeFieldStorage.emplace_back("v_weight_scale", mVWeightScale.data(), PluginFieldType::kFLOAT32,
            static_cast<int32_t>(mVWeightScale.size()));
        mSerializeFieldStorage.emplace_back("q_bias", mQBias.data(), PluginFieldType::kFLOAT32,
            static_cast<int32_t>(mQBias.size()));
        mSerializeFieldStorage.emplace_back("k_bias", mKBias.data(), PluginFieldType::kFLOAT32,
            static_cast<int32_t>(mKBias.size()));
        mSerializeFieldStorage.emplace_back("v_bias", mVBias.data(), PluginFieldType::kFLOAT32,
            static_cast<int32_t>(mVBias.size()));
        mSerializeFields.nbFields = static_cast<int32_t>(mSerializeFieldStorage.size());
        mSerializeFields.fields = mSerializeFieldStorage.data();
    }

    std::string mName;
    float mQAs{};
    float mKAs{};
    float mVAs{};
    std::vector<float> mQSmooth;
    std::vector<float> mKSmooth;
    std::vector<float> mVSmooth;
    std::vector<int8_t> mQWeight;
    std::vector<int8_t> mKWeight;
    std::vector<int8_t> mVWeight;
    std::vector<float> mQWeightScale;
    std::vector<float> mKWeightScale;
    std::vector<float> mVWeightScale;
    std::vector<float> mQBias;
    std::vector<float> mKBias;
    std::vector<float> mVBias;
    PluginFieldCollection mSerializeFields{};
    std::vector<PluginField> mSerializeFieldStorage;
    cublasLtHandle_t mHandle{};
    LtPlan mQPlan{};
    LtPlan mKvPlan{};
    float* mDQSmooth{};
    float* mDKSmooth{};
    float* mDVSmooth{};
    int8_t* mDQWeight{};
    int8_t* mDKWeight{};
    int8_t* mDVWeight{};
    float* mDQWeightScale{};
    float* mDKWeightScale{};
    float* mDVWeightScale{};
    float* mDQBias{};
    float* mDKBias{};
    float* mDVBias{};
    bool mInitialized{false};
};

class SmoothQuantQkvPluginCreator final : public IPluginCreatorV3One
{
public:
    SmoothQuantQkvPluginCreator()
    {
        mFieldStorage.emplace_back("q_activation_scale", nullptr, PluginFieldType::kFLOAT32, 1);
        mFieldStorage.emplace_back("k_activation_scale", nullptr, PluginFieldType::kFLOAT32, 1);
        mFieldStorage.emplace_back("v_activation_scale", nullptr, PluginFieldType::kFLOAT32, 1);
        mFieldStorage.emplace_back("q_smooth_scale", nullptr, PluginFieldType::kFLOAT32, kK);
        mFieldStorage.emplace_back("k_smooth_scale", nullptr, PluginFieldType::kFLOAT32, kK);
        mFieldStorage.emplace_back("v_smooth_scale", nullptr, PluginFieldType::kFLOAT32, kK);
        mFieldStorage.emplace_back("q_weight", nullptr, PluginFieldType::kINT8, kNQ * kK);
        mFieldStorage.emplace_back("k_weight", nullptr, PluginFieldType::kINT8, kNKV * kK);
        mFieldStorage.emplace_back("v_weight", nullptr, PluginFieldType::kINT8, kNKV * kK);
        mFieldStorage.emplace_back("q_weight_scale", nullptr, PluginFieldType::kFLOAT32, kNQ);
        mFieldStorage.emplace_back("k_weight_scale", nullptr, PluginFieldType::kFLOAT32, kNKV);
        mFieldStorage.emplace_back("v_weight_scale", nullptr, PluginFieldType::kFLOAT32, kNKV);
        mFieldStorage.emplace_back("q_bias", nullptr, PluginFieldType::kFLOAT32, kNQ);
        mFieldStorage.emplace_back("k_bias", nullptr, PluginFieldType::kFLOAT32, kNKV);
        mFieldStorage.emplace_back("v_bias", nullptr, PluginFieldType::kFLOAT32, kNKV);
        mFields.nbFields = static_cast<int32_t>(mFieldStorage.size());
        mFields.fields = mFieldStorage.data();
    }

    IPluginV3* createPlugin(AsciiChar const* name, PluginFieldCollection const* fc, TensorRTPhase) noexcept override
    {
        float qAs = getFloat(fc, "q_activation_scale", mQAs);
        float kAs = getFloat(fc, "k_activation_scale", mKAs);
        float vAs = getFloat(fc, "v_activation_scale", mVAs);
        return new SmoothQuantQkvPlugin(name ? name : "smoothquant_qkv", qAs, kAs, vAs,
            getFloatVector(fc, "q_smooth_scale", mQSmooth, kK),
            getFloatVector(fc, "k_smooth_scale", mKSmooth, kK),
            getFloatVector(fc, "v_smooth_scale", mVSmooth, kK),
            getInt8Vector(fc, "q_weight", mQWeight, static_cast<size_t>(kNQ) * kK),
            getInt8Vector(fc, "k_weight", mKWeight, static_cast<size_t>(kNKV) * kK),
            getInt8Vector(fc, "v_weight", mVWeight, static_cast<size_t>(kNKV) * kK),
            getFloatVector(fc, "q_weight_scale", mQWeightScale, kNQ),
            getFloatVector(fc, "k_weight_scale", mKWeightScale, kNKV),
            getFloatVector(fc, "v_weight_scale", mVWeightScale, kNKV),
            getFloatVector(fc, "q_bias", mQBias, kNQ),
            getFloatVector(fc, "k_bias", mKBias, kNKV),
            getFloatVector(fc, "v_bias", mVBias, kNKV));
    }

    PluginFieldCollection const* getFieldNames() noexcept override { return &mFields; }
    AsciiChar const* getPluginName() const noexcept override { return "SmolVLASmoothQuantQKVSerializedV3Smoke"; }
    AsciiChar const* getPluginVersion() const noexcept override { return "1"; }
    AsciiChar const* getPluginNamespace() const noexcept override { return ""; }

    void setScales(float qAs, float kAs, float vAs)
    {
        mQAs = qAs;
        mKAs = kAs;
        mVAs = vAs;
    }

    void setWeights(std::vector<float> qSmooth, std::vector<float> kSmooth, std::vector<float> vSmooth,
        std::vector<int8_t> qWeight, std::vector<int8_t> kWeight, std::vector<int8_t> vWeight,
        std::vector<float> qWeightScale, std::vector<float> kWeightScale, std::vector<float> vWeightScale,
        std::vector<float> qBias, std::vector<float> kBias, std::vector<float> vBias)
    {
        mQSmooth = std::move(qSmooth);
        mKSmooth = std::move(kSmooth);
        mVSmooth = std::move(vSmooth);
        mQWeight = std::move(qWeight);
        mKWeight = std::move(kWeight);
        mVWeight = std::move(vWeight);
        mQWeightScale = std::move(qWeightScale);
        mKWeightScale = std::move(kWeightScale);
        mVWeightScale = std::move(vWeightScale);
        mQBias = std::move(qBias);
        mKBias = std::move(kBias);
        mVBias = std::move(vBias);
    }

private:
    static PluginField const* findField(PluginFieldCollection const* fc, char const* name)
    {
        if (!fc || !fc->fields)
        {
            return nullptr;
        }
        for (int32_t i = 0; i < fc->nbFields; ++i)
        {
            PluginField const& field = fc->fields[i];
            if (field.name && std::strcmp(field.name, name) == 0)
            {
                return &field;
            }
        }
        return nullptr;
    }

    static float getFloat(PluginFieldCollection const* fc, char const* name, float fallback)
    {
        PluginField const* field = findField(fc, name);
        if (!field || !field->data || field->type != PluginFieldType::kFLOAT32 || field->length < 1)
        {
            return fallback;
        }
        return *static_cast<float const*>(field->data);
    }

    static std::vector<float> getFloatVector(
        PluginFieldCollection const* fc, char const* name, std::vector<float> const& fallback, size_t expected)
    {
        PluginField const* field = findField(fc, name);
        if (!field || !field->data || field->type != PluginFieldType::kFLOAT32
            || static_cast<size_t>(field->length) != expected)
        {
            return fallback;
        }
        auto const* begin = static_cast<float const*>(field->data);
        return std::vector<float>(begin, begin + expected);
    }

    static std::vector<int8_t> getInt8Vector(
        PluginFieldCollection const* fc, char const* name, std::vector<int8_t> const& fallback, size_t expected)
    {
        PluginField const* field = findField(fc, name);
        if (!field || !field->data || field->type != PluginFieldType::kINT8
            || static_cast<size_t>(field->length) != expected)
        {
            return fallback;
        }
        auto const* begin = static_cast<int8_t const*>(field->data);
        return std::vector<int8_t>(begin, begin + expected);
    }

    PluginFieldCollection mFields{};
    std::vector<PluginField> mFieldStorage;
    float mQAs{};
    float mKAs{};
    float mVAs{};
    std::vector<float> mQSmooth;
    std::vector<float> mKSmooth;
    std::vector<float> mVSmooth;
    std::vector<int8_t> mQWeight;
    std::vector<int8_t> mKWeight;
    std::vector<int8_t> mVWeight;
    std::vector<float> mQWeightScale;
    std::vector<float> mKWeightScale;
    std::vector<float> mVWeightScale;
    std::vector<float> mQBias;
    std::vector<float> mKBias;
    std::vector<float> mVBias;
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
    std::filesystem::path outDir = argc > 2 ? argv[2] : "runs/deploy/6/J-qkv-plugin-serialized-network-api";
    float qAs = argc > 3 ? std::atof(argv[3]) : 1.0F;
    float kAs = argc > 4 ? std::atof(argv[4]) : 1.0F;
    float vAs = argc > 5 ? std::atof(argv[5]) : 1.0F;
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

    Logger logger;
    SmoothQuantQkvPluginCreator creator;
    creator.setScales(qAs, kAs, vAs);
    creator.setWeights(hQS, hKS, hVS, hQw, hKw, hVw, hQWs, hKWs, hVWs, hQBias, hKBias, hVBias);
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
    ITensor* inputs[1]{};
    inputs[0] = network->addInput("x", DataType::kFLOAT, Dims3{kBatch, kSeqLen, kK});
    for (ITensor* t : inputs)
    {
        if (!t)
        {
            std::fprintf(stderr, "addInput failed\n");
            return 1;
        }
        t->setAllowedFormats(linear);
    }
    SmoothQuantQkvPlugin plugin{"smoothquant_qkv", qAs, kAs, vAs, hQS, hKS, hVS, hQw, hKw, hVw, hQWs, hKWs,
        hVWs, hQBias, hKBias, hVBias};
    IPluginV3Layer* layer = network->addPluginV3(inputs, 1, nullptr, 0, plugin);
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
    std::filesystem::path planPath = outDir / "smoothquant_qkv_plugin_serialized_smoke.plan";
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

    void* dev[4]{};
    mallocCopy(&dev[0], hX.data(), hX.size() * sizeof(float), "x");
    checkCuda(cudaMalloc(&dev[1], hQRef.size() * sizeof(float)), "q out");
    checkCuda(cudaMalloc(&dev[2], hKRef.size() * sizeof(float)), "k out");
    checkCuda(cudaMalloc(&dev[3], hVRef.size() * sizeof(float)), "v out");

    char const* names[] = {"x", "q_raw", "k_raw", "v_raw"};
    for (int32_t i = 0; i < 4; ++i)
    {
        context->setTensorAddress(names[i], dev[i]);
    }
    cudaStream_t stream{};
    checkCuda(cudaStreamCreate(&stream), "stream");
    float latencyMs = timeCuda(stream, warmup, iters, *context);

    std::vector<float> hQOut(hQRef.size());
    std::vector<float> hKOut(hKRef.size());
    std::vector<float> hVOut(hVRef.size());
    checkCuda(cudaMemcpy(hQOut.data(), dev[1], hQOut.size() * sizeof(float), cudaMemcpyDeviceToHost), "q copy");
    checkCuda(cudaMemcpy(hKOut.data(), dev[2], hKOut.size() * sizeof(float), cudaMemcpyDeviceToHost), "k copy");
    checkCuda(cudaMemcpy(hVOut.data(), dev[3], hVOut.size() * sizeof(float), cudaMemcpyDeviceToHost), "v copy");
    Metrics qm = computeMetrics(hQRef, hQOut);
    Metrics km = computeMetrics(hKRef, hKOut);
    Metrics vm = computeMetrics(hVRef, hVOut);

    std::ofstream report(outDir / "smoothquant_qkv_plugin_serialized_smoke_report.json");
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
