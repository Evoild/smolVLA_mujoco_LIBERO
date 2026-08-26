#include "fused_rope_layout.h"

#include <NvInferPlugin.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstring>

using nvinfer1::DataType;
using nvinfer1::DimsExprs;
using nvinfer1::DynamicPluginTensorDesc;
using nvinfer1::IExprBuilder;
using nvinfer1::IPluginV2;
using nvinfer1::IPluginV2DynamicExt;
using nvinfer1::PluginField;
using nvinfer1::PluginFieldCollection;
using nvinfer1::PluginFieldType;
using nvinfer1::PluginTensorDesc;
using nvinfer1::TensorFormat;

namespace smolvla
{
namespace
{

template <typename T>
__device__ inline float loadScalar(T const* ptr, int64_t idx)
{
    return static_cast<float>(ptr[idx]);
}

template <>
__device__ inline float loadScalar<__half>(__half const* ptr, int64_t idx)
{
    return __half2float(ptr[idx]);
}

template <>
__device__ inline float loadScalar<__nv_bfloat16>(__nv_bfloat16 const* ptr, int64_t idx)
{
    return __bfloat162float(ptr[idx]);
}

template <typename T>
__device__ inline void storeScalar(T* ptr, int64_t idx, float value)
{
    ptr[idx] = static_cast<T>(value);
}

template <>
__device__ inline void storeScalar<__half>(__half* ptr, int64_t idx, float value)
{
    ptr[idx] = __float2half(value);
}

template <>
__device__ inline void storeScalar<__nv_bfloat16>(__nv_bfloat16* ptr, int64_t idx, float value)
{
    ptr[idx] = __float2bfloat16(value);
}

template <typename PosT>
__device__ inline int32_t loadPosition(PosT const* ptr, int64_t idx)
{
    return static_cast<int32_t>(ptr[idx]);
}

__device__ inline float warpReduceSum(float value)
{
    unsigned int mask = 0xffffffffU;
    value += __shfl_down_sync(mask, value, 16);
    value += __shfl_down_sync(mask, value, 8);
    value += __shfl_down_sync(mask, value, 4);
    value += __shfl_down_sync(mask, value, 2);
    value += __shfl_down_sync(mask, value, 1);
    return value;
}

template <typename T, typename PosT>
__global__ void fusedRopeLayoutKernel(T const* qRaw, T const* kRaw, PosT const* positionIds, T* qOut, T* kOut,
    int32_t batch, int32_t seqLen, int32_t qHeads, int32_t kvHeads, int32_t headDim, float maxWavelength)
{
    int64_t total = static_cast<int64_t>(batch) * seqLen * qHeads * headDim;
    int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int32_t halfDim = headDim / 2;
    int32_t groups = qHeads / kvHeads;

    for (; linear < total; linear += static_cast<int64_t>(gridDim.x) * blockDim.x)
    {
        int32_t d = static_cast<int32_t>(linear % headDim);
        int64_t t0 = linear / headDim;
        int32_t qh = static_cast<int32_t>(t0 % qHeads);
        t0 /= qHeads;
        int32_t l = static_cast<int32_t>(t0 % seqLen);
        int32_t b = static_cast<int32_t>(t0 / seqLen);

        int32_t halfIndex = d < halfDim ? d : d - halfDim;
        int32_t otherD = d < halfDim ? d + halfDim : d - halfDim;
        int32_t pos = loadPosition(positionIds, static_cast<int64_t>(b) * seqLen + l);
        float freqExponent = (2.0F / static_cast<float>(headDim)) * static_cast<float>(halfIndex);
        float radians = static_cast<float>(pos) / powf(maxWavelength, freqExponent);
        float s = sinf(radians);
        float c = cosf(radians);

        int64_t qBase = (static_cast<int64_t>(b) * seqLen + l) * qHeads * headDim
            + static_cast<int64_t>(qh) * headDim;
        float qv = loadScalar(qRaw, qBase + d);
        float qOther = loadScalar(qRaw, qBase + otherD);
        float qRot = d < halfDim ? (qv * c - qOther * s) : (qv * c + qOther * s);
        int64_t qOutIdx = ((static_cast<int64_t>(b) * qHeads + qh) * seqLen + l) * headDim + d;
        storeScalar(qOut, qOutIdx, qRot);

        int32_t kvh = qh / groups;
        int64_t kBase = (static_cast<int64_t>(b) * seqLen + l) * kvHeads * headDim
            + static_cast<int64_t>(kvh) * headDim;
        float kv = loadScalar(kRaw, kBase + d);
        float kOther = loadScalar(kRaw, kBase + otherD);
        float kRot = d < halfDim ? (kv * c - kOther * s) : (kv * c + kOther * s);
        int64_t kOutIdx = ((static_cast<int64_t>(b) * qHeads + qh) * headDim + d) * seqLen + l;
        storeScalar(kOut, kOutIdx, kRot);
    }
}

template <typename T, typename PosT>
__global__ void fusedRopeQKKernel(T const* qRaw, T const* kRaw, PosT const* positionIds, float* scores,
    int32_t batch, int32_t seqLen, int32_t qHeads, int32_t kvHeads, int32_t headDim, float maxWavelength)
{
    int64_t total = static_cast<int64_t>(batch) * qHeads * seqLen * seqLen;
    int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int32_t halfDim = headDim / 2;
    int32_t groups = qHeads / kvHeads;

    for (; linear < total; linear += static_cast<int64_t>(gridDim.x) * blockDim.x)
    {
        int32_t kPos = static_cast<int32_t>(linear % seqLen);
        int64_t t0 = linear / seqLen;
        int32_t qPos = static_cast<int32_t>(t0 % seqLen);
        t0 /= seqLen;
        int32_t qh = static_cast<int32_t>(t0 % qHeads);
        int32_t b = static_cast<int32_t>(t0 / qHeads);
        int32_t kvh = qh / groups;

        int32_t qPosition = loadPosition(positionIds, static_cast<int64_t>(b) * seqLen + qPos);
        int32_t kPosition = loadPosition(positionIds, static_cast<int64_t>(b) * seqLen + kPos);
        float acc = 0.0F;
        for (int32_t d = 0; d < headDim; ++d)
        {
            int32_t halfIndex = d < halfDim ? d : d - halfDim;
            int32_t otherD = d < halfDim ? d + halfDim : d - halfDim;
            float freqExponent = (2.0F / static_cast<float>(headDim)) * static_cast<float>(halfIndex);
            float invFreq = 1.0F / powf(maxWavelength, freqExponent);

            int64_t qBase = (static_cast<int64_t>(b) * seqLen + qPos) * qHeads * headDim
                + static_cast<int64_t>(qh) * headDim;
            float qv = loadScalar(qRaw, qBase + d);
            float qOther = loadScalar(qRaw, qBase + otherD);
            float qRadians = static_cast<float>(qPosition) * invFreq;
            float qs = sinf(qRadians);
            float qc = cosf(qRadians);
            float qRot = d < halfDim ? (qv * qc - qOther * qs) : (qv * qc + qOther * qs);

            int64_t kBase = (static_cast<int64_t>(b) * seqLen + kPos) * kvHeads * headDim
                + static_cast<int64_t>(kvh) * headDim;
            float kv = loadScalar(kRaw, kBase + d);
            float kOther = loadScalar(kRaw, kBase + otherD);
            float kRadians = static_cast<float>(kPosition) * invFreq;
            float ks = sinf(kRadians);
            float kc = cosf(kRadians);
            float kRot = d < halfDim ? (kv * kc - kOther * ks) : (kv * kc + kOther * ks);
            acc += qRot * kRot;
        }
        scores[linear] = acc;
    }
}

template <typename T, typename PosT>
__global__ void fusedRopeQKSoftmaxKernel(T const* qRaw, T const* kRaw, PosT const* positionIds, bool const* mask,
    float* probs, int32_t batch, int32_t seqLen, int32_t qHeads, int32_t kvHeads, int32_t headDim,
    float maxWavelength)
{
    extern __shared__ float smem[];
    float* row = smem;
    float* reduce = smem + blockDim.x;
    int32_t rowId = static_cast<int32_t>(blockIdx.x);
    int32_t qPos = rowId % seqLen;
    int32_t h = (rowId / seqLen) % qHeads;
    int32_t b = rowId / (seqLen * qHeads);
    int32_t groups = qHeads / kvHeads;
    int32_t kvh = h / groups;
    int32_t halfDim = headDim / 2;
    float scale = rsqrtf(static_cast<float>(headDim));

    float localMax = -3.4028234663852886e38F;
    for (int32_t kPos = threadIdx.x; kPos < seqLen; kPos += blockDim.x)
    {
        bool keep = mask[((static_cast<int64_t>(b) * seqLen + qPos) * seqLen) + kPos];
        float score = -3.4028234663852886e38F;
        if (keep)
        {
            int32_t qPosition = loadPosition(positionIds, static_cast<int64_t>(b) * seqLen + qPos);
            int32_t kPosition = loadPosition(positionIds, static_cast<int64_t>(b) * seqLen + kPos);
            float acc = 0.0F;
            for (int32_t d = 0; d < headDim; ++d)
            {
                int32_t halfIndex = d < halfDim ? d : d - halfDim;
                int32_t otherD = d < halfDim ? d + halfDim : d - halfDim;
                float freqExponent = (2.0F / static_cast<float>(headDim)) * static_cast<float>(halfIndex);
                float invFreq = 1.0F / powf(maxWavelength, freqExponent);
                int64_t qBase = (static_cast<int64_t>(b) * seqLen + qPos) * qHeads * headDim
                    + static_cast<int64_t>(h) * headDim;
                float qv = loadScalar(qRaw, qBase + d);
                float qOther = loadScalar(qRaw, qBase + otherD);
                float qRadians = static_cast<float>(qPosition) * invFreq;
                float qs = sinf(qRadians);
                float qc = cosf(qRadians);
                float qRot = d < halfDim ? (qv * qc - qOther * qs) : (qv * qc + qOther * qs);
                int64_t kBase = (static_cast<int64_t>(b) * seqLen + kPos) * kvHeads * headDim
                    + static_cast<int64_t>(kvh) * headDim;
                float kv = loadScalar(kRaw, kBase + d);
                float kOther = loadScalar(kRaw, kBase + otherD);
                float kRadians = static_cast<float>(kPosition) * invFreq;
                float ks = sinf(kRadians);
                float kc = cosf(kRadians);
                float kRot = d < halfDim ? (kv * kc - kOther * ks) : (kv * kc + kOther * ks);
                acc += qRot * kRot;
            }
            score = acc * scale;
        }
        row[kPos] = score;
        localMax = fmaxf(localMax, score);
    }
    reduce[threadIdx.x] = localMax;
    __syncthreads();
    for (int32_t stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (threadIdx.x < stride)
        {
            float other = threadIdx.x + stride < seqLen ? row[threadIdx.x + stride] : -3.4028234663852886e38F;
            reduce[threadIdx.x] = fmaxf(reduce[threadIdx.x], other);
        }
        __syncthreads();
    }
    float maxVal = reduce[0];
    float localSum = 0.0F;
    for (int32_t kPos = threadIdx.x; kPos < seqLen; kPos += blockDim.x)
    {
        float v = expf(row[kPos] - maxVal);
        row[kPos] = v;
        localSum += v;
    }
    reduce[threadIdx.x] = localSum;
    __syncthreads();
    for (int32_t stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (threadIdx.x < stride)
        {
            reduce[threadIdx.x] += reduce[threadIdx.x + stride];
        }
        __syncthreads();
    }
    float denom = reduce[0];
    for (int32_t kPos = threadIdx.x; kPos < seqLen; kPos += blockDim.x)
    {
        int64_t outIdx = ((static_cast<int64_t>(b) * qHeads + h) * seqLen + qPos) * seqLen + kPos;
        probs[outIdx] = row[kPos] / denom;
    }
}

template <typename T, typename PosT>
__global__ void fusedRopeAttentionKernel(T const* qRaw, T const* kRaw, T const* vRaw, PosT const* positionIds,
    bool const* mask, float* context, int32_t batch, int32_t seqLen, int32_t qHeads, int32_t kvHeads,
    int32_t headDim, float maxWavelength)
{
    int32_t qPos = blockIdx.x % seqLen;
    int32_t h = (blockIdx.x / seqLen) % qHeads;
    int32_t b = blockIdx.x / (seqLen * qHeads);
    int32_t groups = qHeads / kvHeads;
    int32_t halfDim = headDim / 2;
    int32_t kvh = h / groups;
    float scale = rsqrtf(static_cast<float>(headDim));
    extern __shared__ float shared[];
    float* scores = shared;
    float* qRotShared = scores + seqLen;
    float* invFreqShared = qRotShared + headDim;
    float* stats = invFreqShared + headDim;
    float* reduce = stats + 2;

    int32_t qPosition = loadPosition(positionIds, static_cast<int64_t>(b) * seqLen + qPos);
    int64_t qBase = (static_cast<int64_t>(b) * seqLen + qPos) * qHeads * headDim
        + static_cast<int64_t>(h) * headDim;
    for (int32_t d = threadIdx.x; d < headDim; d += blockDim.x)
    {
        int32_t halfIndex = d < halfDim ? d : d - halfDim;
        int32_t otherD = d < halfDim ? d + halfDim : d - halfDim;
        float freqExponent = (2.0F / static_cast<float>(headDim)) * static_cast<float>(halfIndex);
        invFreqShared[d] = 1.0F / powf(maxWavelength, freqExponent);
        float qRadians = static_cast<float>(qPosition) * invFreqShared[d];
        float qs = sinf(qRadians);
        float qc = cosf(qRadians);
        float qv = loadScalar(qRaw, qBase + d);
        float qOther = loadScalar(qRaw, qBase + otherD);
        qRotShared[d] = d < halfDim ? (qv * qc - qOther * qs) : (qv * qc + qOther * qs);
    }
    __syncthreads();

    int32_t lane = threadIdx.x & 31;
    int32_t warpId = threadIdx.x >> 5;
    int32_t warpsPerBlock = blockDim.x >> 5;
    for (int32_t kPos = warpId; kPos < seqLen; kPos += warpsPerBlock)
    {
        bool keep = mask[((static_cast<int64_t>(b) * seqLen + qPos) * seqLen) + kPos];
        if (!keep)
        {
            if (lane == 0)
            {
                scores[kPos] = -3.4028234663852886e38F;
            }
            continue;
        }
        int32_t kPosition = loadPosition(positionIds, static_cast<int64_t>(b) * seqLen + kPos);
        float acc = 0.0F;
        int64_t kBase = (static_cast<int64_t>(b) * seqLen + kPos) * kvHeads * headDim
            + static_cast<int64_t>(kvh) * headDim;
        for (int32_t d = lane; d < headDim; d += 32)
        {
            int32_t otherD = d < halfDim ? d + halfDim : d - halfDim;
            float kRadians = static_cast<float>(kPosition) * invFreqShared[d];
            float ks = sinf(kRadians);
            float kc = cosf(kRadians);
            float kv = loadScalar(kRaw, kBase + d);
            float kOther = loadScalar(kRaw, kBase + otherD);
            float kRot = d < halfDim ? (kv * kc - kOther * ks) : (kv * kc + kOther * ks);
            acc += qRotShared[d] * kRot;
        }
        acc = warpReduceSum(acc);
        if (lane == 0)
        {
            scores[kPos] = acc * scale;
        }
    }
    __syncthreads();

    float localMax = -3.4028234663852886e38F;
    for (int32_t kPos = threadIdx.x; kPos < seqLen; kPos += blockDim.x)
    {
        if (scores[kPos] > -3.0e38F)
        {
            localMax = fmaxf(localMax, scores[kPos]);
        }
    }
    reduce[threadIdx.x] = localMax;
    __syncthreads();
    for (int32_t stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (threadIdx.x < stride)
        {
            reduce[threadIdx.x] = fmaxf(reduce[threadIdx.x], reduce[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0)
    {
        stats[0] = reduce[0] > -3.0e38F ? 1.0F : 0.0F;
        stats[1] = reduce[0];
    }
    __syncthreads();

    float localDenom = 0.0F;
    if (stats[0] != 0.0F)
    {
        float maxScore = stats[1];
        for (int32_t kPos = threadIdx.x; kPos < seqLen; kPos += blockDim.x)
        {
            if (scores[kPos] > -3.0e38F)
            {
                float prob = expf(scores[kPos] - maxScore);
                scores[kPos] = prob;
                localDenom += prob;
            }
            else
            {
                scores[kPos] = 0.0F;
            }
        }
    }
    reduce[threadIdx.x] = localDenom;
    __syncthreads();
    for (int32_t stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (threadIdx.x < stride)
        {
            reduce[threadIdx.x] += reduce[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0)
    {
        stats[1] = reduce[0];
    }
    __syncthreads();

    for (int32_t dOut = threadIdx.x; dOut < headDim; dOut += blockDim.x)
    {
        float value = 0.0F;
        if (stats[0] == 0.0F)
        {
            for (int32_t kPos = 0; kPos < seqLen; ++kPos)
            {
                int64_t vBase = (static_cast<int64_t>(b) * seqLen + kPos) * kvHeads * headDim
                    + static_cast<int64_t>(kvh) * headDim;
                value += loadScalar(vRaw, vBase + dOut);
            }
            value /= static_cast<float>(seqLen);
        }
        else
        {
            float invDenom = 1.0F / stats[1];
            for (int32_t kPos = 0; kPos < seqLen; ++kPos)
            {
                float prob = scores[kPos] * invDenom;
                int64_t vBase = (static_cast<int64_t>(b) * seqLen + kPos) * kvHeads * headDim
                    + static_cast<int64_t>(kvh) * headDim;
                value += prob * loadScalar(vRaw, vBase + dOut);
            }
        }
        int64_t outIdx = ((static_cast<int64_t>(b) * qHeads + h) * seqLen + qPos) * headDim + dOut;
        context[outIdx] = value;
    }
}

__global__ void fusedRopeAttentionFp32MaskImplicitPosKernel(float const* qRaw, float const* kRaw, float const* vRaw,
    float const* mask, float* context, int32_t batch, int32_t seqLen, int32_t qHeads, int32_t kvHeads,
    int32_t headDim, float maxWavelength)
{
    int64_t total = static_cast<int64_t>(batch) * qHeads * seqLen * headDim;
    int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int32_t groups = qHeads / kvHeads;
    int32_t halfDim = headDim / 2;
    float scale = rsqrtf(static_cast<float>(headDim));
    for (; linear < total; linear += static_cast<int64_t>(gridDim.x) * blockDim.x)
    {
        int32_t dOut = static_cast<int32_t>(linear % headDim);
        int64_t t0 = linear / headDim;
        int32_t qPos = static_cast<int32_t>(t0 % seqLen);
        t0 /= seqLen;
        int32_t h = static_cast<int32_t>(t0 % qHeads);
        int32_t b = static_cast<int32_t>(t0 / qHeads);
        int32_t kvh = h / groups;

        bool anyKeep = false;
        float maxScore = -3.4028234663852886e38F;
        for (int32_t kPos = 0; kPos < seqLen; ++kPos)
        {
            if (mask[(static_cast<int64_t>(b) * seqLen + qPos) * seqLen + kPos] <= 0.5F)
            {
                continue;
            }
            anyKeep = true;
            float acc = 0.0F;
            for (int32_t d = 0; d < headDim; ++d)
            {
                int32_t halfIndex = d < halfDim ? d : d - halfDim;
                int32_t otherD = d < halfDim ? d + halfDim : d - halfDim;
                float freqExponent = (2.0F / static_cast<float>(headDim)) * static_cast<float>(halfIndex);
                float invFreq = 1.0F / powf(maxWavelength, freqExponent);
                int64_t qBase = (static_cast<int64_t>(b) * seqLen + qPos) * qHeads * headDim
                    + static_cast<int64_t>(h) * headDim;
                float qv = qRaw[qBase + d];
                float qOther = qRaw[qBase + otherD];
                float qRadians = static_cast<float>(qPos) * invFreq;
                float qs = sinf(qRadians);
                float qc = cosf(qRadians);
                float qRot = d < halfDim ? (qv * qc - qOther * qs) : (qv * qc + qOther * qs);
                int64_t kBase = (static_cast<int64_t>(b) * seqLen + kPos) * kvHeads * headDim
                    + static_cast<int64_t>(kvh) * headDim;
                float kv = kRaw[kBase + d];
                float kOther = kRaw[kBase + otherD];
                float kRadians = static_cast<float>(kPos) * invFreq;
                float ks = sinf(kRadians);
                float kc = cosf(kRadians);
                float kRot = d < halfDim ? (kv * kc - kOther * ks) : (kv * kc + kOther * ks);
                acc += qRot * kRot;
            }
            maxScore = fmaxf(maxScore, acc * scale);
        }

        if (!anyKeep)
        {
            float value = 0.0F;
            for (int32_t kPos = 0; kPos < seqLen; ++kPos)
            {
                int64_t vBase = (static_cast<int64_t>(b) * seqLen + kPos) * kvHeads * headDim
                    + static_cast<int64_t>(kvh) * headDim;
                value += vRaw[vBase + dOut];
            }
            context[linear] = value / static_cast<float>(seqLen);
            continue;
        }

        float denom = 0.0F;
        float value = 0.0F;
        for (int32_t kPos = 0; kPos < seqLen; ++kPos)
        {
            if (mask[(static_cast<int64_t>(b) * seqLen + qPos) * seqLen + kPos] <= 0.5F)
            {
                continue;
            }
            float acc = 0.0F;
            for (int32_t d = 0; d < headDim; ++d)
            {
                int32_t halfIndex = d < halfDim ? d : d - halfDim;
                int32_t otherD = d < halfDim ? d + halfDim : d - halfDim;
                float freqExponent = (2.0F / static_cast<float>(headDim)) * static_cast<float>(halfIndex);
                float invFreq = 1.0F / powf(maxWavelength, freqExponent);
                int64_t qBase = (static_cast<int64_t>(b) * seqLen + qPos) * qHeads * headDim
                    + static_cast<int64_t>(h) * headDim;
                float qv = qRaw[qBase + d];
                float qOther = qRaw[qBase + otherD];
                float qRadians = static_cast<float>(qPos) * invFreq;
                float qs = sinf(qRadians);
                float qc = cosf(qRadians);
                float qRot = d < halfDim ? (qv * qc - qOther * qs) : (qv * qc + qOther * qs);
                int64_t kBase = (static_cast<int64_t>(b) * seqLen + kPos) * kvHeads * headDim
                    + static_cast<int64_t>(kvh) * headDim;
                float kv = kRaw[kBase + d];
                float kOther = kRaw[kBase + otherD];
                float kRadians = static_cast<float>(kPos) * invFreq;
                float ks = sinf(kRadians);
                float kc = cosf(kRadians);
                float kRot = d < halfDim ? (kv * kc - kOther * ks) : (kv * kc + kOther * ks);
                acc += qRot * kRot;
            }
            float prob = expf(acc * scale - maxScore);
            denom += prob;
            int64_t vBase = (static_cast<int64_t>(b) * seqLen + kPos) * kvHeads * headDim
                + static_cast<int64_t>(kvh) * headDim;
            value += prob * vRaw[vBase + dOut];
        }
        context[linear] = value / denom;
    }
}

template <typename T>
int32_t launchTyped(void const* qRaw, void const* kRaw, void const* positionIds, void* qOut, void* kOut,
    int32_t batch, int32_t seqLen, FusedRopeLayoutParams const& params, DataType positionType, cudaStream_t stream)
{
    int64_t total = static_cast<int64_t>(batch) * seqLen * params.qHeads * params.headDim;
    int32_t block = 256;
    int32_t grid = static_cast<int32_t>(std::min<int64_t>((total + block - 1) / block, 4096));
    if (positionType == DataType::kINT32)
    {
        fusedRopeLayoutKernel<T, int32_t><<<grid, block, 0, stream>>>(static_cast<T const*>(qRaw),
            static_cast<T const*>(kRaw), static_cast<int32_t const*>(positionIds), static_cast<T*>(qOut),
            static_cast<T*>(kOut), batch, seqLen, params.qHeads, params.kvHeads, params.headDim,
            params.maxWavelength);
    }
    else if (positionType == DataType::kINT64)
    {
        fusedRopeLayoutKernel<T, int64_t><<<grid, block, 0, stream>>>(static_cast<T const*>(qRaw),
            static_cast<T const*>(kRaw), static_cast<int64_t const*>(positionIds), static_cast<T*>(qOut),
            static_cast<T*>(kOut), batch, seqLen, params.qHeads, params.kvHeads, params.headDim,
            params.maxWavelength);
    }
    else
    {
        return 1;
    }
    return cudaPeekAtLastError() == cudaSuccess ? 0 : 2;
}

template <typename T>
int32_t launchQKTyped(void const* qRaw, void const* kRaw, void const* positionIds, void* scores, int32_t batch,
    int32_t seqLen, FusedRopeLayoutParams const& params, DataType positionType, cudaStream_t stream)
{
    int64_t total = static_cast<int64_t>(batch) * params.qHeads * seqLen * seqLen;
    int32_t block = 128;
    int32_t grid = static_cast<int32_t>(std::min<int64_t>((total + block - 1) / block, 4096));
    if (positionType == DataType::kINT32)
    {
        fusedRopeQKKernel<T, int32_t><<<grid, block, 0, stream>>>(static_cast<T const*>(qRaw),
            static_cast<T const*>(kRaw), static_cast<int32_t const*>(positionIds), static_cast<float*>(scores), batch,
            seqLen, params.qHeads, params.kvHeads, params.headDim, params.maxWavelength);
    }
    else if (positionType == DataType::kINT64)
    {
        fusedRopeQKKernel<T, int64_t><<<grid, block, 0, stream>>>(static_cast<T const*>(qRaw),
            static_cast<T const*>(kRaw), static_cast<int64_t const*>(positionIds), static_cast<float*>(scores), batch,
            seqLen, params.qHeads, params.kvHeads, params.headDim, params.maxWavelength);
    }
    else
    {
        return 1;
    }
    return cudaPeekAtLastError() == cudaSuccess ? 0 : 2;
}

template <typename T>
int32_t launchQKSoftmaxTyped(void const* qRaw, void const* kRaw, void const* positionIds, void const* mask,
    void* probs, int32_t batch, int32_t seqLen, FusedRopeLayoutParams const& params, DataType positionType,
    cudaStream_t stream)
{
    int32_t block = 256;
    int32_t grid = batch * params.qHeads * seqLen;
    size_t smem = static_cast<size_t>(block) * 2U * sizeof(float);
    if (positionType == DataType::kINT32)
    {
        fusedRopeQKSoftmaxKernel<T, int32_t><<<grid, block, smem, stream>>>(static_cast<T const*>(qRaw),
            static_cast<T const*>(kRaw), static_cast<int32_t const*>(positionIds), static_cast<bool const*>(mask),
            static_cast<float*>(probs), batch, seqLen, params.qHeads, params.kvHeads, params.headDim,
            params.maxWavelength);
    }
    else if (positionType == DataType::kINT64)
    {
        fusedRopeQKSoftmaxKernel<T, int64_t><<<grid, block, smem, stream>>>(static_cast<T const*>(qRaw),
            static_cast<T const*>(kRaw), static_cast<int64_t const*>(positionIds), static_cast<bool const*>(mask),
            static_cast<float*>(probs), batch, seqLen, params.qHeads, params.kvHeads, params.headDim,
            params.maxWavelength);
    }
    else
    {
        return 1;
    }
    return cudaPeekAtLastError() == cudaSuccess ? 0 : 2;
}

template <typename T>
int32_t launchAttentionTyped(void const* qRaw, void const* kRaw, void const* vRaw, void const* positionIds,
    void const* mask, void* context, int32_t batch, int32_t seqLen, FusedRopeLayoutParams const& params,
    DataType positionType, cudaStream_t stream)
{
    int32_t block = 256;
    int32_t grid = batch * params.qHeads * seqLen;
    size_t smem = static_cast<size_t>(seqLen + params.headDim * 2 + 2 + block) * sizeof(float);
    if (positionType == DataType::kINT32)
    {
        fusedRopeAttentionKernel<T, int32_t><<<grid, block, smem, stream>>>(static_cast<T const*>(qRaw),
            static_cast<T const*>(kRaw), static_cast<T const*>(vRaw), static_cast<int32_t const*>(positionIds),
            static_cast<bool const*>(mask), static_cast<float*>(context), batch, seqLen, params.qHeads, params.kvHeads,
            params.headDim, params.maxWavelength);
    }
    else if (positionType == DataType::kINT64)
    {
        fusedRopeAttentionKernel<T, int64_t><<<grid, block, smem, stream>>>(static_cast<T const*>(qRaw),
            static_cast<T const*>(kRaw), static_cast<T const*>(vRaw), static_cast<int64_t const*>(positionIds),
            static_cast<bool const*>(mask), static_cast<float*>(context), batch, seqLen, params.qHeads, params.kvHeads,
            params.headDim, params.maxWavelength);
    }
    else
    {
        return 1;
    }
    return cudaPeekAtLastError() == cudaSuccess ? 0 : 2;
}

template <typename T>
void write(char*& ptr, T const& value)
{
    std::memcpy(ptr, &value, sizeof(T));
    ptr += sizeof(T);
}

template <typename T>
void read(char const*& ptr, T& value)
{
    std::memcpy(&value, ptr, sizeof(T));
    ptr += sizeof(T);
}

int32_t getFieldInt(PluginFieldCollection const* fc, char const* name, int32_t fallback)
{
    if (fc == nullptr)
    {
        return fallback;
    }
    for (int32_t i = 0; i < fc->nbFields; ++i)
    {
        PluginField const& field = fc->fields[i];
        if (std::strcmp(field.name, name) == 0 && field.data != nullptr)
        {
            return *static_cast<int32_t const*>(field.data);
        }
    }
    return fallback;
}

float getFieldFloat(PluginFieldCollection const* fc, char const* name, float fallback)
{
    if (fc == nullptr)
    {
        return fallback;
    }
    for (int32_t i = 0; i < fc->nbFields; ++i)
    {
        PluginField const& field = fc->fields[i];
        if (std::strcmp(field.name, name) == 0 && field.data != nullptr)
        {
            return *static_cast<float const*>(field.data);
        }
    }
    return fallback;
}

} // namespace

int32_t launchFusedRopeLayout(void const* qRaw, void const* kRaw, void const* positionIds, void* qOut, void* kOut,
    int32_t batch, int32_t seqLen, FusedRopeLayoutParams const& params, DataType dataType, DataType positionType,
    cudaStream_t stream)
{
    if (dataType == DataType::kFLOAT)
    {
        return launchTyped<float>(qRaw, kRaw, positionIds, qOut, kOut, batch, seqLen, params, positionType, stream);
    }
    if (dataType == DataType::kHALF)
    {
        return launchTyped<__half>(qRaw, kRaw, positionIds, qOut, kOut, batch, seqLen, params, positionType, stream);
    }
    if (dataType == DataType::kBF16)
    {
        return launchTyped<__nv_bfloat16>(qRaw, kRaw, positionIds, qOut, kOut, batch, seqLen, params, positionType,
            stream);
    }
    return 1;
}

int32_t launchFusedRopeQK(void const* qRaw, void const* kRaw, void const* positionIds, void* scores, int32_t batch,
    int32_t seqLen, FusedRopeLayoutParams const& params, DataType dataType, DataType positionType, cudaStream_t stream)
{
    if (dataType == DataType::kFLOAT)
    {
        return launchQKTyped<float>(qRaw, kRaw, positionIds, scores, batch, seqLen, params, positionType, stream);
    }
    if (dataType == DataType::kHALF)
    {
        return launchQKTyped<__half>(qRaw, kRaw, positionIds, scores, batch, seqLen, params, positionType, stream);
    }
    if (dataType == DataType::kBF16)
    {
        return launchQKTyped<__nv_bfloat16>(qRaw, kRaw, positionIds, scores, batch, seqLen, params, positionType,
            stream);
    }
    return 1;
}

int32_t launchFusedRopeQKSoftmax(void const* qRaw, void const* kRaw, void const* positionIds, void const* mask,
    void* probs, int32_t batch, int32_t seqLen, FusedRopeLayoutParams const& params, DataType dataType,
    DataType positionType, cudaStream_t stream)
{
    if (dataType == DataType::kFLOAT)
    {
        return launchQKSoftmaxTyped<float>(qRaw, kRaw, positionIds, mask, probs, batch, seqLen, params, positionType,
            stream);
    }
    if (dataType == DataType::kHALF)
    {
        return launchQKSoftmaxTyped<__half>(qRaw, kRaw, positionIds, mask, probs, batch, seqLen, params, positionType,
            stream);
    }
    if (dataType == DataType::kBF16)
    {
        return launchQKSoftmaxTyped<__nv_bfloat16>(qRaw, kRaw, positionIds, mask, probs, batch, seqLen, params,
            positionType, stream);
    }
    return 1;
}

int32_t launchFusedRopeAttention(void const* qRaw, void const* kRaw, void const* vRaw, void const* positionIds,
    void const* mask, void* context, int32_t batch, int32_t seqLen, FusedRopeLayoutParams const& params,
    DataType dataType, DataType positionType, cudaStream_t stream)
{
    if (dataType == DataType::kFLOAT)
    {
        return launchAttentionTyped<float>(qRaw, kRaw, vRaw, positionIds, mask, context, batch, seqLen, params,
            positionType, stream);
    }
    if (dataType == DataType::kHALF)
    {
        return launchAttentionTyped<__half>(qRaw, kRaw, vRaw, positionIds, mask, context, batch, seqLen, params,
            positionType, stream);
    }
    if (dataType == DataType::kBF16)
    {
        return launchAttentionTyped<__nv_bfloat16>(qRaw, kRaw, vRaw, positionIds, mask, context, batch, seqLen, params,
            positionType, stream);
    }
    return 1;
}

int32_t launchFusedRopeAttentionFp32MaskImplicitPos(void const* qRaw, void const* kRaw, void const* vRaw,
    void const* mask, void* context, int32_t batch, int32_t seqLen, FusedRopeLayoutParams const& params,
    cudaStream_t stream)
{
    int64_t total = static_cast<int64_t>(batch) * params.qHeads * seqLen * params.headDim;
    int32_t block = 128;
    int32_t grid = static_cast<int32_t>(std::min<int64_t>((total + block - 1) / block, 4096));
    fusedRopeAttentionFp32MaskImplicitPosKernel<<<grid, block, 0, stream>>>(static_cast<float const*>(qRaw),
        static_cast<float const*>(kRaw), static_cast<float const*>(vRaw), static_cast<float const*>(mask),
        static_cast<float*>(context), batch, seqLen, params.qHeads, params.kvHeads, params.headDim,
        params.maxWavelength);
    return cudaPeekAtLastError() == cudaSuccess ? 0 : 2;
}

FusedRopeLayoutPlugin::FusedRopeLayoutPlugin(char const* name, FusedRopeLayoutParams params)
    : mName(name)
    , mParams(params)
{
}

FusedRopeLayoutPlugin::FusedRopeLayoutPlugin(char const* name, void const* serialData, size_t serialLength)
    : mName(name)
{
    if (serialLength == sizeof(FusedRopeLayoutParams))
    {
        char const* ptr = static_cast<char const*>(serialData);
        read(ptr, mParams);
    }
}

IPluginV2DynamicExt* FusedRopeLayoutPlugin::clone() const noexcept
{
    auto* plugin = new FusedRopeLayoutPlugin(mName.c_str(), mParams);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

DimsExprs FusedRopeLayoutPlugin::getOutputDimensions(
    int32_t outputIndex, DimsExprs const* inputs, int32_t nbInputs, IExprBuilder& exprBuilder) noexcept
{
    DimsExprs out{};
    out.nbDims = 4;
    out.d[0] = inputs[0].d[0];
    if (outputIndex == 0)
    {
        out.d[1] = exprBuilder.constant(mParams.qHeads);
        out.d[2] = inputs[0].d[1];
        out.d[3] = exprBuilder.constant(mParams.headDim);
    }
    else
    {
        out.d[1] = exprBuilder.constant(mParams.qHeads);
        out.d[2] = exprBuilder.constant(mParams.headDim);
        out.d[3] = inputs[0].d[1];
    }
    (void) nbInputs;
    return out;
}

bool FusedRopeLayoutPlugin::supportsFormatCombination(
    int32_t pos, PluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    (void) nbInputs;
    (void) nbOutputs;
    if (inOut[pos].format != TensorFormat::kLINEAR)
    {
        return false;
    }
    if (pos == 2)
    {
        return inOut[pos].type == DataType::kINT32 || inOut[pos].type == DataType::kINT64;
    }
    if (pos == 0)
    {
        return inOut[pos].type == DataType::kFLOAT || inOut[pos].type == DataType::kHALF
            || inOut[pos].type == DataType::kBF16;
    }
    return inOut[pos].type == inOut[0].type;
}

void FusedRopeLayoutPlugin::configurePlugin(
    DynamicPluginTensorDesc const* in, int32_t nbInputs, DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept
{
    (void) in;
    (void) nbInputs;
    (void) out;
    (void) nbOutputs;
}

size_t FusedRopeLayoutPlugin::getWorkspaceSize(
    PluginTensorDesc const*, int32_t, PluginTensorDesc const*, int32_t) const noexcept
{
    return 0;
}

int32_t FusedRopeLayoutPlugin::enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const*,
    void const* const* inputs, void* const* outputs, void*, cudaStream_t stream) noexcept
{
    int32_t batch = inputDesc[0].dims.d[0];
    int32_t seqLen = inputDesc[0].dims.d[1];
    return launchFusedRopeLayout(inputs[0], inputs[1], inputs[2], outputs[0], outputs[1], batch, seqLen, mParams,
        inputDesc[0].type, inputDesc[2].type, stream);
}

int32_t FusedRopeLayoutPlugin::initialize() noexcept
{
    return 0;
}

void FusedRopeLayoutPlugin::terminate() noexcept {}

size_t FusedRopeLayoutPlugin::getSerializationSize() const noexcept
{
    return sizeof(FusedRopeLayoutParams);
}

void FusedRopeLayoutPlugin::serialize(void* buffer) const noexcept
{
    char* ptr = static_cast<char*>(buffer);
    write(ptr, mParams);
}

void FusedRopeLayoutPlugin::destroy() noexcept
{
    delete this;
}

void FusedRopeLayoutPlugin::setPluginNamespace(char const* pluginNamespace) noexcept
{
    mNamespace = pluginNamespace ? pluginNamespace : "";
}

char const* FusedRopeLayoutPlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

char const* FusedRopeLayoutPlugin::getPluginType() const noexcept
{
    return kFusedRopeLayoutPluginName;
}

char const* FusedRopeLayoutPlugin::getPluginVersion() const noexcept
{
    return kFusedRopeLayoutPluginVersion;
}

int32_t FusedRopeLayoutPlugin::getNbOutputs() const noexcept
{
    return 2;
}

DataType FusedRopeLayoutPlugin::getOutputDataType(int32_t, DataType const* inputTypes, int32_t) const noexcept
{
    return inputTypes[0];
}

void FusedRopeLayoutPlugin::attachToContext(
    cudnnContext*, cublasContext*, nvinfer1::IGpuAllocator*) noexcept
{
}

void FusedRopeLayoutPlugin::detachFromContext() noexcept {}

FusedRopeLayoutPluginCreator::FusedRopeLayoutPluginCreator()
{
    mFields.emplace_back("q_heads", nullptr, PluginFieldType::kINT32, 1);
    mFields.emplace_back("kv_heads", nullptr, PluginFieldType::kINT32, 1);
    mFields.emplace_back("head_dim", nullptr, PluginFieldType::kINT32, 1);
    mFields.emplace_back("max_wavelength", nullptr, PluginFieldType::kFLOAT32, 1);
    mFieldCollection.nbFields = static_cast<int32_t>(mFields.size());
    mFieldCollection.fields = mFields.data();
}

char const* FusedRopeLayoutPluginCreator::getPluginName() const noexcept
{
    return kFusedRopeLayoutPluginName;
}

char const* FusedRopeLayoutPluginCreator::getPluginVersion() const noexcept
{
    return kFusedRopeLayoutPluginVersion;
}

PluginFieldCollection const* FusedRopeLayoutPluginCreator::getFieldNames() noexcept
{
    return &mFieldCollection;
}

IPluginV2* FusedRopeLayoutPluginCreator::createPlugin(char const* name, PluginFieldCollection const* fc) noexcept
{
    FusedRopeLayoutParams params{};
    params.qHeads = getFieldInt(fc, "q_heads", params.qHeads);
    params.kvHeads = getFieldInt(fc, "kv_heads", params.kvHeads);
    params.headDim = getFieldInt(fc, "head_dim", params.headDim);
    params.maxWavelength = getFieldFloat(fc, "max_wavelength", params.maxWavelength);
    return new FusedRopeLayoutPlugin(name, params);
}

IPluginV2* FusedRopeLayoutPluginCreator::deserializePlugin(
    char const* name, void const* serialData, size_t serialLength) noexcept
{
    return new FusedRopeLayoutPlugin(name, serialData, serialLength);
}

void FusedRopeLayoutPluginCreator::setPluginNamespace(char const* pluginNamespace) noexcept
{
    mNamespace = pluginNamespace ? pluginNamespace : "";
}

char const* FusedRopeLayoutPluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

FusedRopeQKPlugin::FusedRopeQKPlugin(char const* name, FusedRopeLayoutParams params)
    : mName(name)
    , mParams(params)
{
}

FusedRopeQKPlugin::FusedRopeQKPlugin(char const* name, void const* serialData, size_t serialLength)
    : mName(name)
{
    if (serialLength == sizeof(FusedRopeLayoutParams))
    {
        char const* ptr = static_cast<char const*>(serialData);
        read(ptr, mParams);
    }
}

IPluginV2DynamicExt* FusedRopeQKPlugin::clone() const noexcept
{
    auto* plugin = new FusedRopeQKPlugin(mName.c_str(), mParams);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

DimsExprs FusedRopeQKPlugin::getOutputDimensions(
    int32_t, DimsExprs const* inputs, int32_t nbInputs, IExprBuilder& exprBuilder) noexcept
{
    DimsExprs out{};
    out.nbDims = 4;
    out.d[0] = inputs[0].d[0];
    out.d[1] = exprBuilder.constant(mParams.qHeads);
    out.d[2] = inputs[0].d[1];
    out.d[3] = inputs[0].d[1];
    (void) nbInputs;
    return out;
}

bool FusedRopeQKPlugin::supportsFormatCombination(
    int32_t pos, PluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    (void) nbInputs;
    (void) nbOutputs;
    if (inOut[pos].format != TensorFormat::kLINEAR)
    {
        return false;
    }
    if (pos == 2)
    {
        return inOut[pos].type == DataType::kINT32 || inOut[pos].type == DataType::kINT64;
    }
    if (pos == 3)
    {
        return inOut[pos].type == DataType::kFLOAT;
    }
    if (pos == 0)
    {
        return inOut[pos].type == DataType::kFLOAT || inOut[pos].type == DataType::kHALF
            || inOut[pos].type == DataType::kBF16;
    }
    return inOut[pos].type == inOut[0].type;
}

void FusedRopeQKPlugin::configurePlugin(
    DynamicPluginTensorDesc const*, int32_t, DynamicPluginTensorDesc const*, int32_t) noexcept
{
}

size_t FusedRopeQKPlugin::getWorkspaceSize(
    PluginTensorDesc const*, int32_t, PluginTensorDesc const*, int32_t) const noexcept
{
    return 0;
}

int32_t FusedRopeQKPlugin::enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const*,
    void const* const* inputs, void* const* outputs, void*, cudaStream_t stream) noexcept
{
    int32_t batch = inputDesc[0].dims.d[0];
    int32_t seqLen = inputDesc[0].dims.d[1];
    return launchFusedRopeQK(inputs[0], inputs[1], inputs[2], outputs[0], batch, seqLen, mParams, inputDesc[0].type,
        inputDesc[2].type, stream);
}

int32_t FusedRopeQKPlugin::initialize() noexcept
{
    return 0;
}

void FusedRopeQKPlugin::terminate() noexcept {}

size_t FusedRopeQKPlugin::getSerializationSize() const noexcept
{
    return sizeof(FusedRopeLayoutParams);
}

void FusedRopeQKPlugin::serialize(void* buffer) const noexcept
{
    char* ptr = static_cast<char*>(buffer);
    write(ptr, mParams);
}

void FusedRopeQKPlugin::destroy() noexcept
{
    delete this;
}

void FusedRopeQKPlugin::setPluginNamespace(char const* pluginNamespace) noexcept
{
    mNamespace = pluginNamespace ? pluginNamespace : "";
}

char const* FusedRopeQKPlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

char const* FusedRopeQKPlugin::getPluginType() const noexcept
{
    return kFusedRopeQKPluginName;
}

char const* FusedRopeQKPlugin::getPluginVersion() const noexcept
{
    return kFusedRopeQKPluginVersion;
}

int32_t FusedRopeQKPlugin::getNbOutputs() const noexcept
{
    return 1;
}

DataType FusedRopeQKPlugin::getOutputDataType(int32_t, DataType const*, int32_t) const noexcept
{
    return DataType::kFLOAT;
}

void FusedRopeQKPlugin::attachToContext(cudnnContext*, cublasContext*, nvinfer1::IGpuAllocator*) noexcept {}

void FusedRopeQKPlugin::detachFromContext() noexcept {}

FusedRopeQKPluginCreator::FusedRopeQKPluginCreator()
{
    mFields.emplace_back("q_heads", nullptr, PluginFieldType::kINT32, 1);
    mFields.emplace_back("kv_heads", nullptr, PluginFieldType::kINT32, 1);
    mFields.emplace_back("head_dim", nullptr, PluginFieldType::kINT32, 1);
    mFields.emplace_back("max_wavelength", nullptr, PluginFieldType::kFLOAT32, 1);
    mFieldCollection.nbFields = static_cast<int32_t>(mFields.size());
    mFieldCollection.fields = mFields.data();
}

char const* FusedRopeQKPluginCreator::getPluginName() const noexcept
{
    return kFusedRopeQKPluginName;
}

char const* FusedRopeQKPluginCreator::getPluginVersion() const noexcept
{
    return kFusedRopeQKPluginVersion;
}

PluginFieldCollection const* FusedRopeQKPluginCreator::getFieldNames() noexcept
{
    return &mFieldCollection;
}

IPluginV2* FusedRopeQKPluginCreator::createPlugin(char const* name, PluginFieldCollection const* fc) noexcept
{
    FusedRopeLayoutParams params{};
    params.qHeads = getFieldInt(fc, "q_heads", params.qHeads);
    params.kvHeads = getFieldInt(fc, "kv_heads", params.kvHeads);
    params.headDim = getFieldInt(fc, "head_dim", params.headDim);
    params.maxWavelength = getFieldFloat(fc, "max_wavelength", params.maxWavelength);
    return new FusedRopeQKPlugin(name, params);
}

IPluginV2* FusedRopeQKPluginCreator::deserializePlugin(
    char const* name, void const* serialData, size_t serialLength) noexcept
{
    return new FusedRopeQKPlugin(name, serialData, serialLength);
}

void FusedRopeQKPluginCreator::setPluginNamespace(char const* pluginNamespace) noexcept
{
    mNamespace = pluginNamespace ? pluginNamespace : "";
}

char const* FusedRopeQKPluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

FusedRopeQKSoftmaxPlugin::FusedRopeQKSoftmaxPlugin(char const* name, FusedRopeLayoutParams params)
    : mName(name)
    , mParams(params)
{
}

FusedRopeQKSoftmaxPlugin::FusedRopeQKSoftmaxPlugin(char const* name, void const* serialData, size_t serialLength)
    : mName(name)
{
    if (serialLength == sizeof(FusedRopeLayoutParams))
    {
        char const* ptr = static_cast<char const*>(serialData);
        read(ptr, mParams);
    }
}

IPluginV2DynamicExt* FusedRopeQKSoftmaxPlugin::clone() const noexcept
{
    auto* plugin = new FusedRopeQKSoftmaxPlugin(mName.c_str(), mParams);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

DimsExprs FusedRopeQKSoftmaxPlugin::getOutputDimensions(
    int32_t, DimsExprs const* inputs, int32_t nbInputs, IExprBuilder& exprBuilder) noexcept
{
    DimsExprs out{};
    out.nbDims = 4;
    out.d[0] = inputs[0].d[0];
    out.d[1] = exprBuilder.constant(mParams.qHeads);
    out.d[2] = inputs[0].d[1];
    out.d[3] = inputs[0].d[1];
    (void) nbInputs;
    return out;
}

bool FusedRopeQKSoftmaxPlugin::supportsFormatCombination(
    int32_t pos, PluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    (void) nbInputs;
    (void) nbOutputs;
    if (inOut[pos].format != TensorFormat::kLINEAR)
    {
        return false;
    }
    if (pos == 2)
    {
        return inOut[pos].type == DataType::kINT32 || inOut[pos].type == DataType::kINT64;
    }
    if (pos == 3)
    {
        return inOut[pos].type == DataType::kBOOL;
    }
    if (pos == 4)
    {
        return inOut[pos].type == DataType::kFLOAT;
    }
    if (pos == 0)
    {
        return inOut[pos].type == DataType::kFLOAT || inOut[pos].type == DataType::kHALF
            || inOut[pos].type == DataType::kBF16;
    }
    return inOut[pos].type == inOut[0].type;
}

void FusedRopeQKSoftmaxPlugin::configurePlugin(
    DynamicPluginTensorDesc const*, int32_t, DynamicPluginTensorDesc const*, int32_t) noexcept
{
}

size_t FusedRopeQKSoftmaxPlugin::getWorkspaceSize(
    PluginTensorDesc const*, int32_t, PluginTensorDesc const*, int32_t) const noexcept
{
    return 0;
}

int32_t FusedRopeQKSoftmaxPlugin::enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const*,
    void const* const* inputs, void* const* outputs, void*, cudaStream_t stream) noexcept
{
    int32_t batch = inputDesc[0].dims.d[0];
    int32_t seqLen = inputDesc[0].dims.d[1];
    return launchFusedRopeQKSoftmax(inputs[0], inputs[1], inputs[2], inputs[3], outputs[0], batch, seqLen, mParams,
        inputDesc[0].type, inputDesc[2].type, stream);
}

int32_t FusedRopeQKSoftmaxPlugin::initialize() noexcept
{
    return 0;
}

void FusedRopeQKSoftmaxPlugin::terminate() noexcept {}

size_t FusedRopeQKSoftmaxPlugin::getSerializationSize() const noexcept
{
    return sizeof(FusedRopeLayoutParams);
}

void FusedRopeQKSoftmaxPlugin::serialize(void* buffer) const noexcept
{
    char* ptr = static_cast<char*>(buffer);
    write(ptr, mParams);
}

void FusedRopeQKSoftmaxPlugin::destroy() noexcept
{
    delete this;
}

void FusedRopeQKSoftmaxPlugin::setPluginNamespace(char const* pluginNamespace) noexcept
{
    mNamespace = pluginNamespace ? pluginNamespace : "";
}

char const* FusedRopeQKSoftmaxPlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

char const* FusedRopeQKSoftmaxPlugin::getPluginType() const noexcept
{
    return kFusedRopeQKSoftmaxPluginName;
}

char const* FusedRopeQKSoftmaxPlugin::getPluginVersion() const noexcept
{
    return kFusedRopeQKSoftmaxPluginVersion;
}

int32_t FusedRopeQKSoftmaxPlugin::getNbOutputs() const noexcept
{
    return 1;
}

DataType FusedRopeQKSoftmaxPlugin::getOutputDataType(int32_t, DataType const*, int32_t) const noexcept
{
    return DataType::kFLOAT;
}

void FusedRopeQKSoftmaxPlugin::attachToContext(cudnnContext*, cublasContext*, nvinfer1::IGpuAllocator*) noexcept {}

void FusedRopeQKSoftmaxPlugin::detachFromContext() noexcept {}

FusedRopeQKSoftmaxPluginCreator::FusedRopeQKSoftmaxPluginCreator()
{
    mFields.emplace_back("q_heads", nullptr, PluginFieldType::kINT32, 1);
    mFields.emplace_back("kv_heads", nullptr, PluginFieldType::kINT32, 1);
    mFields.emplace_back("head_dim", nullptr, PluginFieldType::kINT32, 1);
    mFields.emplace_back("max_wavelength", nullptr, PluginFieldType::kFLOAT32, 1);
    mFieldCollection.nbFields = static_cast<int32_t>(mFields.size());
    mFieldCollection.fields = mFields.data();
}

char const* FusedRopeQKSoftmaxPluginCreator::getPluginName() const noexcept
{
    return kFusedRopeQKSoftmaxPluginName;
}

char const* FusedRopeQKSoftmaxPluginCreator::getPluginVersion() const noexcept
{
    return kFusedRopeQKSoftmaxPluginVersion;
}

PluginFieldCollection const* FusedRopeQKSoftmaxPluginCreator::getFieldNames() noexcept
{
    return &mFieldCollection;
}

IPluginV2* FusedRopeQKSoftmaxPluginCreator::createPlugin(char const* name, PluginFieldCollection const* fc) noexcept
{
    FusedRopeLayoutParams params{};
    params.qHeads = getFieldInt(fc, "q_heads", params.qHeads);
    params.kvHeads = getFieldInt(fc, "kv_heads", params.kvHeads);
    params.headDim = getFieldInt(fc, "head_dim", params.headDim);
    params.maxWavelength = getFieldFloat(fc, "max_wavelength", params.maxWavelength);
    return new FusedRopeQKSoftmaxPlugin(name, params);
}

IPluginV2* FusedRopeQKSoftmaxPluginCreator::deserializePlugin(
    char const* name, void const* serialData, size_t serialLength) noexcept
{
    return new FusedRopeQKSoftmaxPlugin(name, serialData, serialLength);
}

void FusedRopeQKSoftmaxPluginCreator::setPluginNamespace(char const* pluginNamespace) noexcept
{
    mNamespace = pluginNamespace ? pluginNamespace : "";
}

char const* FusedRopeQKSoftmaxPluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

FusedRopeAttentionPlugin::FusedRopeAttentionPlugin(char const* name, FusedRopeLayoutParams params)
    : mName(name)
    , mParams(params)
{
}

FusedRopeAttentionPlugin::FusedRopeAttentionPlugin(char const* name, void const* serialData, size_t serialLength)
    : mName(name)
{
    if (serialLength == sizeof(FusedRopeLayoutParams))
    {
        char const* ptr = static_cast<char const*>(serialData);
        read(ptr, mParams);
    }
}

IPluginV2DynamicExt* FusedRopeAttentionPlugin::clone() const noexcept
{
    auto* plugin = new FusedRopeAttentionPlugin(mName.c_str(), mParams);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

DimsExprs FusedRopeAttentionPlugin::getOutputDimensions(
    int32_t, DimsExprs const* inputs, int32_t nbInputs, IExprBuilder& exprBuilder) noexcept
{
    DimsExprs out{};
    out.nbDims = 4;
    out.d[0] = inputs[0].d[0];
    out.d[1] = exprBuilder.constant(mParams.qHeads);
    out.d[2] = inputs[0].d[1];
    out.d[3] = exprBuilder.constant(mParams.headDim);
    (void) nbInputs;
    return out;
}

bool FusedRopeAttentionPlugin::supportsFormatCombination(
    int32_t pos, PluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    (void) nbInputs;
    (void) nbOutputs;
    if (inOut[pos].format != TensorFormat::kLINEAR)
    {
        return false;
    }
    if (pos == 3)
    {
        return inOut[pos].type == DataType::kINT32 || inOut[pos].type == DataType::kINT64;
    }
    if (pos == 4)
    {
        return inOut[pos].type == DataType::kBOOL;
    }
    if (pos == 5)
    {
        return inOut[pos].type == DataType::kFLOAT;
    }
    if (pos == 0)
    {
        return inOut[pos].type == DataType::kFLOAT || inOut[pos].type == DataType::kHALF
            || inOut[pos].type == DataType::kBF16;
    }
    return inOut[pos].type == inOut[0].type;
}

void FusedRopeAttentionPlugin::configurePlugin(
    DynamicPluginTensorDesc const*, int32_t, DynamicPluginTensorDesc const*, int32_t) noexcept
{
}

size_t FusedRopeAttentionPlugin::getWorkspaceSize(
    PluginTensorDesc const*, int32_t, PluginTensorDesc const*, int32_t) const noexcept
{
    return 0;
}

int32_t FusedRopeAttentionPlugin::enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const*,
    void const* const* inputs, void* const* outputs, void*, cudaStream_t stream) noexcept
{
    int32_t batch = inputDesc[0].dims.d[0];
    int32_t seqLen = inputDesc[0].dims.d[1];
    return launchFusedRopeAttention(inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], outputs[0], batch, seqLen,
        mParams, inputDesc[0].type, inputDesc[3].type, stream);
}

int32_t FusedRopeAttentionPlugin::initialize() noexcept { return 0; }
void FusedRopeAttentionPlugin::terminate() noexcept {}
size_t FusedRopeAttentionPlugin::getSerializationSize() const noexcept { return sizeof(FusedRopeLayoutParams); }
void FusedRopeAttentionPlugin::serialize(void* buffer) const noexcept
{
    char* ptr = static_cast<char*>(buffer);
    write(ptr, mParams);
}
void FusedRopeAttentionPlugin::destroy() noexcept { delete this; }
void FusedRopeAttentionPlugin::setPluginNamespace(char const* pluginNamespace) noexcept
{
    mNamespace = pluginNamespace ? pluginNamespace : "";
}
char const* FusedRopeAttentionPlugin::getPluginNamespace() const noexcept { return mNamespace.c_str(); }
char const* FusedRopeAttentionPlugin::getPluginType() const noexcept { return kFusedRopeAttentionPluginName; }
char const* FusedRopeAttentionPlugin::getPluginVersion() const noexcept { return kFusedRopeAttentionPluginVersion; }
int32_t FusedRopeAttentionPlugin::getNbOutputs() const noexcept { return 1; }
DataType FusedRopeAttentionPlugin::getOutputDataType(int32_t, DataType const*, int32_t) const noexcept
{
    return DataType::kFLOAT;
}
void FusedRopeAttentionPlugin::attachToContext(cudnnContext*, cublasContext*, nvinfer1::IGpuAllocator*) noexcept {}
void FusedRopeAttentionPlugin::detachFromContext() noexcept {}

FusedRopeAttentionPluginCreator::FusedRopeAttentionPluginCreator()
{
    mFields.emplace_back("q_heads", nullptr, PluginFieldType::kINT32, 1);
    mFields.emplace_back("kv_heads", nullptr, PluginFieldType::kINT32, 1);
    mFields.emplace_back("head_dim", nullptr, PluginFieldType::kINT32, 1);
    mFields.emplace_back("max_wavelength", nullptr, PluginFieldType::kFLOAT32, 1);
    mFieldCollection.nbFields = static_cast<int32_t>(mFields.size());
    mFieldCollection.fields = mFields.data();
}
char const* FusedRopeAttentionPluginCreator::getPluginName() const noexcept { return kFusedRopeAttentionPluginName; }
char const* FusedRopeAttentionPluginCreator::getPluginVersion() const noexcept { return kFusedRopeAttentionPluginVersion; }
PluginFieldCollection const* FusedRopeAttentionPluginCreator::getFieldNames() noexcept { return &mFieldCollection; }
IPluginV2* FusedRopeAttentionPluginCreator::createPlugin(char const* name, PluginFieldCollection const* fc) noexcept
{
    FusedRopeLayoutParams params{};
    params.qHeads = getFieldInt(fc, "q_heads", params.qHeads);
    params.kvHeads = getFieldInt(fc, "kv_heads", params.kvHeads);
    params.headDim = getFieldInt(fc, "head_dim", params.headDim);
    params.maxWavelength = getFieldFloat(fc, "max_wavelength", params.maxWavelength);
    return new FusedRopeAttentionPlugin(name, params);
}
IPluginV2* FusedRopeAttentionPluginCreator::deserializePlugin(
    char const* name, void const* serialData, size_t serialLength) noexcept
{
    return new FusedRopeAttentionPlugin(name, serialData, serialLength);
}
void FusedRopeAttentionPluginCreator::setPluginNamespace(char const* pluginNamespace) noexcept
{
    mNamespace = pluginNamespace ? pluginNamespace : "";
}
char const* FusedRopeAttentionPluginCreator::getPluginNamespace() const noexcept { return mNamespace.c_str(); }

} // namespace smolvla

using SmolVLAFusedRopeLayoutPluginCreator = smolvla::FusedRopeLayoutPluginCreator;
using SmolVLAFusedRopeQKPluginCreator = smolvla::FusedRopeQKPluginCreator;
using SmolVLAFusedRopeQKSoftmaxPluginCreator = smolvla::FusedRopeQKSoftmaxPluginCreator;
using SmolVLAFusedRopeAttentionPluginCreator = smolvla::FusedRopeAttentionPluginCreator;
REGISTER_TENSORRT_PLUGIN(SmolVLAFusedRopeLayoutPluginCreator);
REGISTER_TENSORRT_PLUGIN(SmolVLAFusedRopeQKPluginCreator);
REGISTER_TENSORRT_PLUGIN(SmolVLAFusedRopeQKSoftmaxPluginCreator);
REGISTER_TENSORRT_PLUGIN(SmolVLAFusedRopeAttentionPluginCreator);
