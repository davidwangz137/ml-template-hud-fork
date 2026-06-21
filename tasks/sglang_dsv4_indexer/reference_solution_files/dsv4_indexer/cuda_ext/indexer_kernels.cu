#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <stdint.h>

namespace {

constexpr float kFP8Max = 448.0f;

template <typename T> __device__ inline float cvt_f(T x) { return (float)x; }
template <> __device__ inline float cvt_f<__nv_bfloat16>(__nv_bfloat16 x) { return __bfloat162float(x); }
template <> __device__ inline float cvt_f<__half>(__half x) { return __half2float(x); }

__device__ inline float warp_max(float v) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) v = fmaxf(v, __shfl_down_sync(0xffffffff, v, offset));
  return v;
}

template <int RowsPerBlock>
__device__ inline float block_max_128(float v, int local_tid, int subrow) {
  __shared__ float partial[RowsPerBlock][4];
  int lane = local_tid & 31;
  int wid = local_tid >> 5;
  v = warp_max(v);
  if (lane == 0) partial[subrow][wid] = v;
  __syncthreads();
  v = local_tid < 4 ? partial[subrow][lane] : 0.0f;
  if (wid == 0) v = warp_max(v);
  return v;
}

template <typename T, int RowsPerBlock>
__global__ void indexer_kernel(const T* __restrict__ q_input, const T* __restrict__ weight,
                               __nv_fp8_e4m3* __restrict__ q_fp8, float* __restrict__ weights_out,
                               float weight_scale, const float* __restrict__ freqs,
                               const int32_t* __restrict__ positions, int rows, int heads) {
  int subrow = threadIdx.x >> 7;
  int tid = threadIdx.x & 127;
  int row = blockIdx.x * RowsPerBlock + subrow;
  if (row >= rows) return;
  int batch = row / heads;
  int pos = positions[batch];
  const T* in = q_input + (int64_t)row * 128;

  __shared__ float vals[RowsPerBlock][128];
  float v = cvt_f<T>(in[tid]);

  if (tid >= 64) {
    int rel = tid - 64;
    int pair = rel >> 1;
    bool imag = rel & 1;
    float mate = cvt_f<T>(in[tid ^ 1]);
    float real = imag ? mate : v;
    float im = imag ? v : mate;
    float fr = freqs[(int64_t)pos * 64 + pair * 2];
    float fi = freqs[(int64_t)pos * 64 + pair * 2 + 1];
    v = imag ? (real * fi + im * fr) : (real * fr - im * fi);
  }
  vals[subrow][tid] = v;
  __syncthreads();

  int lane = tid >> 2;
  int elem = tid & 3;
  int base = lane * 4;
  float d0 = vals[subrow][base + 0];
  float d1 = vals[subrow][base + 1];
  float d2 = vals[subrow][base + 2];
  float d3 = vals[subrow][base + 3];
  float a0 = d0 + d1;
  float a1 = d0 - d1;
  float a2 = d2 + d3;
  float a3 = d2 - d3;
  d0 = a0 + a2;
  d1 = a1 + a3;
  d2 = a0 - a2;
  d3 = a1 - a3;
  float data = elem == 0 ? d0 : (elem == 1 ? d1 : (elem == 2 ? d2 : d3));

#pragma unroll
  for (int mask = 1; mask < 32; mask <<= 1) {
    vals[subrow][tid] = data;
    __syncthreads();
    float other = vals[subrow][((lane ^ mask) << 2) + elem];
    data = (lane & mask) ? (other - data) : (data + other);
    __syncthreads();
  }
  data *= rsqrtf(128.0f);

  float abs_v = fabsf(data);
  float max_v = block_max_128<RowsPerBlock>(abs_v, tid, subrow);
  __shared__ float scale_s[RowsPerBlock];
  if (tid == 0) scale_s[subrow] = fmaxf(1.0e-4f, max_v) / kFP8Max;
  __syncthreads();
  float scaled = fminf(fmaxf(data / scale_s[subrow], -kFP8Max), kFP8Max);
  q_fp8[(int64_t)row * 128 + tid] = __nv_fp8_e4m3(scaled);
  if (tid == 0) weights_out[row] = cvt_f<T>(weight[row]) * weight_scale * scale_s[subrow];
}

template <typename T>
void launch_t(const void* q_input, const void* weight, void* q_fp8, float* weights_out,
              float weight_scale, const float* freqs, const int32_t* positions,
              int rows, int heads, cudaStream_t stream) {
  constexpr int kRowsPerBlock = 4;
  int blocks = (rows + kRowsPerBlock - 1) / kRowsPerBlock;
  indexer_kernel<T, kRowsPerBlock><<<blocks, kRowsPerBlock * 128, 0, stream>>>((const T*)q_input, (const T*)weight,
                                                                               (__nv_fp8_e4m3*)q_fp8, weights_out,
                                                                               weight_scale, freqs, positions, rows, heads);
}

} // namespace

void launch_indexer(const void* q_input, const void* weight, void* q_fp8, float* weights_out,
                    float weight_scale, const float* freqs, const int32_t* positions,
                    int rows, int heads, int dtype, cudaStream_t stream) {
  if (dtype == 0) launch_t<__nv_bfloat16>(q_input, weight, q_fp8, weights_out, weight_scale, freqs, positions, rows, heads, stream);
  else launch_t<__half>(q_input, weight, q_fp8, weights_out, weight_scale, freqs, positions, rows, heads, stream);
}
