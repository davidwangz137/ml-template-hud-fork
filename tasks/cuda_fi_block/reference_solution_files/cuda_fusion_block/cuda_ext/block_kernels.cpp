#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime_api.h>

void launch_rmsnorm(const void*, const void*, void*, int, int, float, int, cudaStream_t);
void launch_fused_add_rmsnorm(void*, void*, const void*, int, int, float, int, cudaStream_t);
void launch_rope(void*, void*, const float*, const float*, int, int, int, int, int, cudaStream_t);
void launch_silu_mul(const void*, void*, int, int, int, cudaStream_t);

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be CUDA")
#define CHECK_CONTIG(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_DTYPE_MATCH(a,b) TORCH_CHECK(a.scalar_type() == b.scalar_type(), #a " and " #b " dtypes differ")

static int dtype_id(at::ScalarType t) {
  if (t == at::ScalarType::BFloat16) return 0;
  if (t == at::ScalarType::Half) return 1;
  if (t == at::ScalarType::Float) return 2;
  TORCH_CHECK(false, "unsupported dtype");
}

at::Tensor rmsnorm(at::Tensor input, at::Tensor weight, double eps) {
  CHECK_CUDA(input); CHECK_CUDA(weight); CHECK_CONTIG(input); CHECK_CONTIG(weight); CHECK_DTYPE_MATCH(input, weight);
  TORCH_CHECK(input.dim() == 2 && weight.dim() == 1);
  int rows = (int)input.size(0), cols = (int)input.size(1);
  TORCH_CHECK(weight.size(0) == cols);
  auto out = at::empty_like(input);
  c10::cuda::CUDAGuard guard(input.device());
  launch_rmsnorm(input.data_ptr(), weight.data_ptr(), out.data_ptr(), rows, cols, (float)eps, dtype_id(input.scalar_type()), at::cuda::getCurrentCUDAStream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

void fused_add_rmsnorm_(at::Tensor input, at::Tensor residual, at::Tensor weight, double eps) {
  CHECK_CUDA(input); CHECK_CUDA(residual); CHECK_CUDA(weight); CHECK_CONTIG(input); CHECK_CONTIG(residual); CHECK_CONTIG(weight);
  CHECK_DTYPE_MATCH(input, residual); CHECK_DTYPE_MATCH(input, weight);
  TORCH_CHECK(input.dim() == 2 && residual.dim() == 2 && weight.dim() == 1);
  TORCH_CHECK(input.sizes() == residual.sizes());
  int rows = (int)input.size(0), cols = (int)input.size(1);
  TORCH_CHECK(weight.size(0) == cols);
  c10::cuda::CUDAGuard guard(input.device());
  launch_fused_add_rmsnorm(input.data_ptr(), residual.data_ptr(), weight.data_ptr(), rows, cols, (float)eps, dtype_id(input.scalar_type()), at::cuda::getCurrentCUDAStream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void apply_rope_(at::Tensor q, at::Tensor k, at::Tensor cos, at::Tensor sin) {
  CHECK_CUDA(q); CHECK_CUDA(k); CHECK_CUDA(cos); CHECK_CUDA(sin); CHECK_CONTIG(q); CHECK_CONTIG(k); CHECK_CONTIG(cos); CHECK_CONTIG(sin);
  CHECK_DTYPE_MATCH(q, k);
  TORCH_CHECK(cos.scalar_type() == at::ScalarType::Float && sin.scalar_type() == at::ScalarType::Float);
  TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && q.size(0) == 1 && k.size(0) == 1);
  int seq = (int)q.size(1), qh = (int)q.size(2), dim = (int)q.size(3);
  int kseq = (int)k.size(1), kh = (int)k.size(2), kdim = (int)k.size(3);
  TORCH_CHECK(kseq == seq && kdim == dim && cos.size(0) == seq && sin.size(0) == seq && cos.size(1) == dim/2 && sin.size(1) == dim/2);
  c10::cuda::CUDAGuard guard(q.device());
  launch_rope(q.data_ptr(), k.data_ptr(), cos.data_ptr<float>(), sin.data_ptr<float>(), seq, qh, kh, dim, dtype_id(q.scalar_type()), at::cuda::getCurrentCUDAStream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

at::Tensor silu_and_mul(at::Tensor gate_up, at::Tensor out) {
  CHECK_CUDA(gate_up); CHECK_CUDA(out); CHECK_CONTIG(gate_up); CHECK_CONTIG(out); CHECK_DTYPE_MATCH(gate_up, out);
  TORCH_CHECK(gate_up.dim() == 2 && out.dim() == 2);
  int rows = (int)gate_up.size(0), hidden = (int)(gate_up.size(1) / 2);
  TORCH_CHECK(gate_up.size(1) == hidden * 2 && out.size(0) == rows && out.size(1) == hidden);
  c10::cuda::CUDAGuard guard(gate_up.device());
  launch_silu_mul(gate_up.data_ptr(), out.data_ptr(), rows, hidden, dtype_id(gate_up.scalar_type()), at::cuda::getCurrentCUDAStream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rmsnorm", &rmsnorm, "RMSNorm forward (CUDA)");
  m.def("fused_add_rmsnorm_", &fused_add_rmsnorm_, "residual add + RMSNorm in-place (CUDA)");
  m.def("apply_rope_", &apply_rope_, "RoPE in-place for q/k NHD tensors (CUDA)");
  m.def("silu_and_mul", &silu_and_mul, "SwiGLU silu(gate)*up (CUDA)");
}
