"""Initial row-wise fused kernel: correctness first, without autotuning."""
import torch
import triton
import triton.language as tl

from .reference import validate_inputs


@triton.jit
def _residual_rmsnorm_kernel(
    X, RESIDUAL, WEIGHT, OUTPUT, RESIDUAL_OUT,
    H: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    columns = tl.arange(0, BLOCK)
    valid = columns < H
    offsets = row * H + columns

    x = tl.load(X + offsets, mask=valid, other=0).to(tl.float32)
    residual = tl.load(RESIDUAL + offsets, mask=valid, other=0).to(tl.float32)

    # Preserve the contract's rounding boundary while retaining values on chip.
    rounded_sum = tl.cast(x + residual, X.dtype.element_ty, fp_downcast_rounding='rtne')
    values = rounded_sum.to(tl.float32)
    mean_square = tl.sum(values * values, axis=0) / H
    inverse_rms = tl.rsqrt(mean_square + EPS)
    weight = tl.load(WEIGHT + columns, mask=valid, other=0).to(tl.float32)
    result = (values * inverse_rms) * weight
    result = tl.cast(result, OUTPUT.dtype.element_ty, fp_downcast_rounding='rtne')

    tl.store(OUTPUT + offsets, result, mask=valid)
    tl.store(RESIDUAL_OUT + offsets, rounded_sum, mask=valid)


def launch_config(hidden_size):
    """Initial fixed policy; no performance-based tuning has taken place."""
    if hidden_size < 1 or hidden_size > 8192:
        raise ValueError('initial kernel supports 1 <= H <= 8192')
    return {'block_size': max(32, triton.next_power_of_2(hidden_size)),
            'num_warps': 4 if hidden_size <= 4096 else 8,
            'num_stages': 1, 'enable_fp_fusion': False}


def residual_rmsnorm_triton_unchecked(x, residual, weight, eps=1e-6):
    """Allocation plus launch; caller validates inputs outside future timing."""
    m, h = x.shape
    config = launch_config(h)
    output = torch.empty_like(x)
    residual_out = torch.empty_like(residual)
    with torch.cuda.device(x.device):
        _residual_rmsnorm_kernel[(m,)](
            x, residual, weight, output, residual_out,
            H=h, EPS=eps, BLOCK=config['block_size'],
            num_warps=config['num_warps'], num_stages=config['num_stages'],
            enable_fp_fusion=config['enable_fp_fusion'],
        )
    return output, residual_out


def residual_rmsnorm_triton(x, residual, weight, eps=1e-6):
    """Validated inference entry point; no input mutation or backward support."""
    validate_inputs(x, residual, weight, eps)
    if torch.is_grad_enabled() and any(t.requires_grad for t in (x, residual, weight)):
        raise RuntimeError('This custom kernel supports forward inference only')
    return residual_rmsnorm_triton_unchecked(x, residual, weight, eps)
