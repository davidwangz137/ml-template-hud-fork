#include <torch/extension.h>

std::vector<at::Tensor> indexer(at::Tensor q_input, at::Tensor weight, double weight_scale,
                                at::Tensor freqs, at::Tensor positions) {
  TORCH_CHECK(false, "Stub file for the DSV4 indexer HUD task: implement CUDA extension");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("indexer", &indexer, "DSV4 indexer RoPE+Hadamard+FP8 quant (CUDA)");
}
