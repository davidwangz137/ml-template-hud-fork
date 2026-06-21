#include <torch/extension.h>

torch::Tensor rmsnorm_cuda(torch::Tensor input, torch::Tensor weight, double eps);
void fused_add_rmsnorm_cuda(torch::Tensor input, torch::Tensor residual, torch::Tensor weight, double eps);
void apply_rope_cuda(torch::Tensor q, torch::Tensor k, torch::Tensor cos, torch::Tensor sin);
torch::Tensor silu_and_mul_cuda(torch::Tensor gate_up, torch::Tensor out);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rmsnorm", &rmsnorm_cuda, "RMSNorm forward");
  m.def("fused_add_rmsnorm_", &fused_add_rmsnorm_cuda, "fused residual add RMSNorm");
  m.def("apply_rope_", &apply_rope_cuda, "in-place RoPE for q/k");
  m.def("silu_and_mul", &silu_and_mul_cuda, "SwiGLU silu-and-mul");
}
