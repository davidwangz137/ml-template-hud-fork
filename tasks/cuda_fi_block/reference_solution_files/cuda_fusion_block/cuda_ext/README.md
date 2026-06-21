# CUDA block extension workspace

Export these functions:

- `rmsnorm(input, weight, eps) -> out`
- `fused_add_rmsnorm_(input, residual, weight, eps) -> None`
- `apply_rope_(q, k, cos, sin) -> None`
- `silu_and_mul(gate_up, out) -> out`

The candidate path may use PyTorch GEMMs and PyTorch SDPA attention. It must not import or call FlashInfer.
