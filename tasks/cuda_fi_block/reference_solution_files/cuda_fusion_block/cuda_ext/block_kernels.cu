#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>

namespace {

inline int h_next_pow2(int x) { int p=1; while(p<x) p<<=1; return p; }

__inline__ __device__ float warp_reduce_sum(float v) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) v += __shfl_down_sync(0xffffffff, v, offset);
  return v;
}

__inline__ __device__ float block_reduce_sum(float v) {
  __shared__ float shared[32];
  int lane = threadIdx.x & 31;
  int wid = threadIdx.x >> 5;
  v = warp_reduce_sum(v);
  if (lane == 0) shared[wid] = v;
  __syncthreads();
  v = (threadIdx.x < ((blockDim.x + 31) >> 5)) ? shared[lane] : 0.f;
  if (wid == 0) v = warp_reduce_sum(v);
  return v;
}

template<typename T> __device__ inline float cvt_f(T x) { return (float)x; }
template<> __device__ inline float cvt_f<__nv_bfloat16>(__nv_bfloat16 x) { return __bfloat162float(x); }
template<> __device__ inline float cvt_f<__half>(__half x) { return __half2float(x); }
template<typename T> __device__ inline T cvt_t(float x) { return (T)x; }
template<> __device__ inline __nv_bfloat16 cvt_t<__nv_bfloat16>(float x) { return __float2bfloat16(x); }
template<> __device__ inline __half cvt_t<__half>(float x) { return __float2half(x); }

template <typename T>
__global__ void rmsnorm_kernel(const T* __restrict__ input, const T* __restrict__ weight,
                               T* __restrict__ output, int rows, int cols, float eps) {
  int row = blockIdx.x;
  const T* x = input + (int64_t)row * cols;
  float ss = 0.f;
  for (int i = threadIdx.x; i < cols; i += blockDim.x) { float v = cvt_f<T>(x[i]); ss += v * v; }
  ss = block_reduce_sum(ss);
  __shared__ float inv;
  if (threadIdx.x == 0) inv = rsqrtf(ss / (float)cols + eps);
  __syncthreads();
  T* y = output + (int64_t)row * cols;
  for (int i = threadIdx.x; i < cols; i += blockDim.x) y[i] = cvt_t<T>(cvt_f<T>(x[i]) * inv * cvt_f<T>(weight[i]));
}

template <typename T>
__global__ void fused_add_rmsnorm_kernel(T* __restrict__ input, T* __restrict__ residual,
                                         const T* __restrict__ weight, int rows, int cols, float eps) {
  int row = blockIdx.x;
  T* in = input + (int64_t)row * cols;
  T* res = residual + (int64_t)row * cols;
  float ss = 0.f;
  for (int i = threadIdx.x; i < cols; i += blockDim.x) {
    float v = cvt_f<T>(res[i]) + cvt_f<T>(in[i]);
    T rv = cvt_t<T>(v);
    res[i] = rv;
    float vr = cvt_f<T>(rv);
    ss += vr * vr;
  }
  ss = block_reduce_sum(ss);
  __shared__ float inv;
  if (threadIdx.x == 0) inv = rsqrtf(ss / (float)cols + eps);
  __syncthreads();
  for (int i = threadIdx.x; i < cols; i += blockDim.x) input[(int64_t)row * cols + i] = cvt_t<T>(cvt_f<T>(res[i]) * inv * cvt_f<T>(weight[i]));
}

template <typename T>
__global__ void rope_one_kernel(T* __restrict__ x, const float* __restrict__ cos, const float* __restrict__ sin,
                                int seq, int heads, int dim) {
  int half = dim >> 1;
  int n = seq * heads * half;
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += blockDim.x * gridDim.x) {
    int i = idx % half;
    int tmp = idx / half;
    int h = tmp % heads;
    int sidx = tmp / heads;
    int base = (sidx * heads + h) * dim;
    float a = cvt_f<T>(x[base + i]);
    float b = cvt_f<T>(x[base + half + i]);
    float c = cos[sidx * half + i];
    float sn = sin[sidx * half + i];
    x[base + i] = cvt_t<T>(a * c - b * sn);
    x[base + half + i] = cvt_t<T>(a * sn + b * c);
  }
}

template <typename T>
__global__ void silu_mul_kernel(const T* __restrict__ gate_up, T* __restrict__ out, int rows, int hidden) {
  int64_t n = (int64_t)rows * hidden;
  for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += (int64_t)blockDim.x * gridDim.x) {
    int col = idx % hidden;
    int row = idx / hidden;
    const T* p = gate_up + (int64_t)row * hidden * 2;
    float g = cvt_f<T>(p[col]);
    float u = cvt_f<T>(p[hidden + col]);
    out[idx] = cvt_t<T>((g / (1.0f + __expf(-g))) * u);
  }
}

template <typename T> inline void lr(const void* input,const void* weight,void* output,int rows,int cols,float eps,cudaStream_t stream){
  int threads = h_next_pow2(cols); if (threads < 32) threads = 32; if (threads > 1024) threads = 1024;
  rmsnorm_kernel<T><<<rows, threads, 0, stream>>>((const T*)input,(const T*)weight,(T*)output,rows,cols,eps);
}
template <typename T> inline void lfar(void* input,void* residual,const void* weight,int rows,int cols,float eps,cudaStream_t stream){
  int threads = h_next_pow2(cols); if (threads < 32) threads = 32; if (threads > 1024) threads = 1024;
  fused_add_rmsnorm_kernel<T><<<rows, threads, 0, stream>>>((T*)input,(T*)residual,(const T*)weight,rows,cols,eps);
}
template <typename T> inline void lrope(void* q,void* k,const float* cos,const float* sin,int seq,int qh,int kh,int dim,cudaStream_t stream){
  int threads=256;
  int qb=(seq*qh*(dim/2)+threads-1)/threads; if(qb>4096) qb=4096;
  int kb=(seq*kh*(dim/2)+threads-1)/threads; if(kb>4096) kb=4096;
  rope_one_kernel<T><<<qb,threads,0,stream>>>((T*)q,cos,sin,seq,qh,dim);
  rope_one_kernel<T><<<kb,threads,0,stream>>>((T*)k,cos,sin,seq,kh,dim);
}
template <typename T> inline void lsm(const void* gate_up,void* out,int rows,int hidden,cudaStream_t stream){
  int threads=256; int64_t blocks=((int64_t)rows*hidden+threads-1)/threads; if(blocks>4096) blocks=4096;
  silu_mul_kernel<T><<<blocks,threads,0,stream>>>((const T*)gate_up,(T*)out,rows,hidden);
}

} // namespace

void launch_rmsnorm(const void* input, const void* weight, void* output, int rows, int cols, float eps, int dtype, cudaStream_t stream) {
  if (dtype == 0) lr<__nv_bfloat16>(input, weight, output, rows, cols, eps, stream);
  else if (dtype == 1) lr<__half>(input, weight, output, rows, cols, eps, stream);
  else lr<float>(input, weight, output, rows, cols, eps, stream);
}
void launch_fused_add_rmsnorm(void* input, void* residual, const void* weight, int rows, int cols, float eps, int dtype, cudaStream_t stream) {
  if (dtype == 0) lfar<__nv_bfloat16>(input, residual, weight, rows, cols, eps, stream);
  else if (dtype == 1) lfar<__half>(input, residual, weight, rows, cols, eps, stream);
  else lfar<float>(input, residual, weight, rows, cols, eps, stream);
}
void launch_rope(void* q, void* k, const float* cos, const float* sin, int seq, int qh, int kh, int dim, int dtype, cudaStream_t stream) {
  if (dtype == 0) lrope<__nv_bfloat16>(q, k, cos, sin, seq, qh, kh, dim, stream);
  else if (dtype == 1) lrope<__half>(q, k, cos, sin, seq, qh, kh, dim, stream);
  else lrope<float>(q, k, cos, sin, seq, qh, kh, dim, stream);
}
void launch_silu_mul(const void* gate_up, void* out, int rows, int hidden, int dtype, cudaStream_t stream) {
  if (dtype == 0) lsm<__nv_bfloat16>(gate_up, out, rows, hidden, stream);
  else if (dtype == 1) lsm<__half>(gate_up, out, rows, hidden, stream);
  else lsm<float>(gate_up, out, rows, hidden, stream);
}
