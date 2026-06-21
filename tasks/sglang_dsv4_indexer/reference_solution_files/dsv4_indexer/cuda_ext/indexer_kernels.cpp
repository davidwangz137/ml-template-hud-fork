#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime_api.h>
#include <torch/extension.h>

void launch_indexer(const void* q_input, const void* weight, void* q_fp8, float* weights_out,
                    float weight_scale, const float* freqs, const int32_t* positions,
                    int rows, int heads, int dtype, cudaStream_t stream);

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be CUDA")
#define CHECK_CONTIG(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

static int dtype_id(at::ScalarType t) {
  if (t == at::ScalarType::BFloat16) return 0;
  if (t == at::ScalarType::Half) return 1;
  TORCH_CHECK(false, "unsupported dtype; expected bf16 or fp16");
}

at::TensorOptions fp8_options_like(const at::Tensor& input) {
  return input.options().dtype(at::ScalarType::Float8_e4m3fn);
}

std::vector<at::Tensor> indexer(at::Tensor q_input, at::Tensor weight, double weight_scale,
                                at::Tensor freqs, at::Tensor positions) {
  CHECK_CUDA(q_input); CHECK_CUDA(weight); CHECK_CUDA(freqs); CHECK_CUDA(positions);
  CHECK_CONTIG(q_input); CHECK_CONTIG(weight); CHECK_CONTIG(freqs); CHECK_CONTIG(positions);
  TORCH_CHECK(q_input.dim() == 3 && q_input.size(2) == 128, "q_input must be [B,H,128]");
  TORCH_CHECK(weight.dim() == 2 && weight.size(0) == q_input.size(0) && weight.size(1) == q_input.size(1));
  TORCH_CHECK(freqs.dim() == 2 && freqs.size(1) == 64 && freqs.scalar_type() == at::ScalarType::Float);
  TORCH_CHECK(positions.dim() == 1 && positions.size(0) == q_input.size(0) && positions.scalar_type() == at::ScalarType::Int);
  TORCH_CHECK(weight.scalar_type() == q_input.scalar_type(), "weight dtype must match q_input");

  auto q_fp8 = at::empty(q_input.sizes(), fp8_options_like(q_input));
  auto weights_out = at::empty({q_input.size(0), q_input.size(1), 1}, q_input.options().dtype(at::ScalarType::Float));
  c10::cuda::CUDAGuard guard(q_input.device());
  launch_indexer(q_input.data_ptr(), weight.data_ptr(), q_fp8.data_ptr(), weights_out.data_ptr<float>(),
                 static_cast<float>(weight_scale), freqs.data_ptr<float>(), positions.data_ptr<int32_t>(),
                 static_cast<int>(q_input.size(0) * q_input.size(1)), static_cast<int>(q_input.size(1)),
                 dtype_id(q_input.scalar_type()), at::cuda::getCurrentCUDAStream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {q_fp8, weights_out};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("indexer", &indexer, "DSV4 indexer RoPE+Hadamard+FP8 quant (CUDA)");
}
