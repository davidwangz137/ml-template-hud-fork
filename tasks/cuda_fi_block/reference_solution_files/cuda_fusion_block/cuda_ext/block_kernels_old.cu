#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <torch/extension.h>

namespace {
constexpr int kThreads = 256;

__device__ __forceinline__ float bf16_to_float(const __nv_bfloat16 x) { return __bfloat162float(x); }
__device__ __forceinline__ __nv_bfloat16 float_to_bf16(const float x) { return __float2bfloat16(x); }

void check_bf16_cuda(const torch::Tensor& t, const char* name) {
  TORCH_CHECK(t.is_cuda(), name, " must be CUDA");
  TORCH_CHECK(t.scalar_type() == torch::kBFloat16, name, " must be bfloat16");
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

__global__ void rmsnorm_kernel(const __nv_bfloat16* __restrict__ input, const __nv_bfloat16* __restrict__ weight, __nv_bfloat16* __restrict__ out, int rows, int dim, float eps) {
  int row = blockIdx.x;
  int base = row * dim;
  float sum = 0.0f;
  for (int col = threadIdx.x; col < dim; col += blockDim.x) {
    float v = bf16_to_float(input[base + col]);
    sum += v * v;
  }
  __shared__ float scratch[kThreads];
  scratch[threadIdx.x] = sum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) scratch[threadIdx.x] += scratch[threadIdx.x + stride];
    __syncthreads();
  }
  float inv = rsqrtf(scratch[0] / static_cast<float>(dim) + eps);
  for (int col = threadIdx.x; col < dim; col += blockDim.x) {
    int idx = base + col;
    out[idx] = float_to_bf16(bf16_to_float(input[idx]) * inv * bf16_to_float(weight[col]));
  }
}

__global__ void fused_add_rmsnorm_kernel(__nv_bfloat16* __restrict__ input, __nv_bfloat16* __restrict__ residual, const __nv_bfloat16* __restrict__ weight, int rows, int dim, float eps) {
  int row = blockIdx.x;
  int base = row * dim;
  float sum = 0.0f;
  for (int col = threadIdx.x; col < dim; col += blockDim.x) {
    int idx = base + col;
    float v = bf16_to_float(residual[idx]) + bf16_to_float(input[idx]);
    residual[idx] = float_to_bf16(v);
    sum += v * v;
  }
  __shared__ float scratch[kThreads];
  scratch[threadIdx.x] = sum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) scratch[threadIdx.x] += scratch[threadIdx.x + stride];
    __syncthreads();
  }
  float inv = rsqrtf(scratch[0] / static_cast<float>(dim) + eps);
  for (int col = threadIdx.x; col < dim; col += blockDim.x) {
    int idx = base + col;
    input[idx] = float_to_bf16(bf16_to_float(residual[idx]) * inv * bf16_to_float(weight[col]));
  }
}

__global__ void rope_kernel(__nv_bfloat16* __restrict__ t, const float* __restrict__ cos, const float* __restrict__ sin, int seq, int heads, int head_dim) {
  int half = head_dim / 2;
  long n = static_cast<long>(seq) * heads * half;
  for (long linear = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x; linear < n; linear += static_cast<long>(blockDim.x) * gridDim.x) {
    int d = static_cast<int>(linear % half);
    long tmp = linear / half;
    int h = static_cast<int>(tmp % heads);
    int sidx = static_cast<int>(tmp / heads);
    long base = ((static_cast<long>(sidx) * heads + h) * head_dim);
    float a = bf16_to_float(t[base + d]);
    float b = bf16_to_float(t[base + d + half]);
    float c = cos[static_cast<long>(sidx) * half + d];
    float ss = sin[static_cast<long>(sidx) * half + d];
    t[base + d] = float_to_bf16(a * c - b * ss);
    t[base + d + half] = float_to_bf16(a * ss + b * c);
  }
}

__global__ void silu_and_mul_kernel(const __nv_bfloat16* __restrict__ gate_up, __nv_bfloat16* __restrict__ out, int rows, int hidden_dim) {
  long n = static_cast<long>(rows) * hidden_dim;
  for (long linear = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x; linear < n; linear += static_cast<long>(blockDim.x) * gridDim.x) {
    int row = static_cast<int>(linear / hidden_dim);
    int col = static_cast<int>(linear - static_cast<long>(row) * hidden_dim);
    long gate_idx = static_cast<long>(row) * hidden_dim * 2 + col;
    float gate = bf16_to_float(gate_up[gate_idx]);
    float up = bf16_to_float(gate_up[gate_idx + hidden_dim]);
    float sig = 1.0f / (1.0f + expf(-gate));
    out[linear] = float_to_bf16(gate * sig * up);
  }
}

}  // namespace

torch::Tensor rmsnorm_cuda(torch::Tensor input, torch::Tensor weight, double eps) {
  check_bf16_cuda(input, "input");
  check_bf16_cuda(weight, "weight");
  TORCH_CHECK(input.dim() == 2, "input must be 2D");
  TORCH_CHECK(weight.dim() == 1 && weight.size(0) == input.size(1), "weight shape mismatch");
  auto out = torch::empty_like(input);
  int rows = static_cast<int>(input.size(0));
  int dim = static_cast<int>(input.size(1));
  rmsnorm_kernel<<<rows, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(input.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr<at::BFloat16>()),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()), rows, dim, static_cast<float>(eps));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

void fused_add_rmsnorm_cuda(torch::Tensor input, torch::Tensor residual, torch::Tensor weight, double eps) {
  check_bf16_cuda(input, "input");
  check_bf16_cuda(residual, "residual");
  check_bf16_cuda(weight, "weight");
  TORCH_CHECK(input.dim() == 2 && residual.dim() == 2 && input.sizes() == residual.sizes(), "shape mismatch");
  int rows = static_cast<int>(input.size(0));
  int dim = static_cast<int>(input.size(1));
  fused_add_rmsnorm_kernel<<<rows, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<__nv_bfloat16*>(input.data_ptr<at::BFloat16>()),
      reinterpret_cast<__nv_bfloat16*>(residual.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr<at::BFloat16>()), rows, dim, static_cast<float>(eps));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void apply_rope_cuda(torch::Tensor q, torch::Tensor k, torch::Tensor cos, torch::Tensor sin) {
  check_bf16_cuda(q, "q");
  check_bf16_cuda(k, "k");
  TORCH_CHECK(cos.is_cuda() && sin.is_cuda() && cos.scalar_type() == torch::kFloat32 && sin.scalar_type() == torch::kFloat32, "cos/sin must be CUDA fp32");
  TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && q.size(0) == 1 && k.size(0) == 1, "q/k must be [1, seq, heads, head_dim]");
  int seq = static_cast<int>(q.size(1));
  int q_heads = static_cast<int>(q.size(2));
  int kv_heads = static_cast<int>(k.size(2));
  int head_dim = static_cast<int>(q.size(3));
  int half = head_dim / 2;
  TORCH_CHECK(k.size(1) == seq && k.size(3) == head_dim && cos.size(0) == seq && cos.size(1) == half && sin.sizes() == cos.sizes(), "rope shape mismatch");
  int q_blocks = static_cast<int>(std::min<long>((static_cast<long>(seq) * q_heads * half + kThreads - 1) / kThreads, 4096));
  int k_blocks = static_cast<int>(std::min<long>((static_cast<long>(seq) * kv_heads * half + kThreads - 1) / kThreads, 4096));
  auto stream = at::cuda::getCurrentCUDAStream();
  rope_kernel<<<q_blocks, kThreads, 0, stream>>>(reinterpret_cast<__nv_bfloat16*>(q.data_ptr<at::BFloat16>()), cos.data_ptr<float>(), sin.data_ptr<float>(), seq, q_heads, head_dim);
  rope_kernel<<<k_blocks, kThreads, 0, stream>>>(reinterpret_cast<__nv_bfloat16*>(k.data_ptr<at::BFloat16>()), cos.data_ptr<float>(), sin.data_ptr<float>(), seq, kv_heads, head_dim);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

torch::Tensor silu_and_mul_cuda(torch::Tensor gate_up, torch::Tensor out) {
  check_bf16_cuda(gate_up, "gate_up");
  check_bf16_cuda(out, "out");
  int rows = static_cast<int>(gate_up.size(0));
  int hidden_dim = static_cast<int>(gate_up.size(1) / 2);
  TORCH_CHECK(out.size(0) == rows && out.size(1) == hidden_dim, "out shape mismatch");
  long n = static_cast<long>(rows) * hidden_dim;
  int blocks = static_cast<int>(std::min<long>((n + kThreads - 1) / kThreads, 4096));
  silu_and_mul_kernel<<<blocks, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(gate_up.data_ptr<at::BFloat16>()),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()), rows, hidden_dim);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
