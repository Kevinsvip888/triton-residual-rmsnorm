"""Thread-count variants of the unchanged initial Triton kernel."""
import torch
from .reference import validate_inputs
from .triton_kernel import _residual_rmsnorm_kernel, launch_config

WARP_CANDIDATES = (1, 2, 4, 8, 16)


def launch_variant_unchecked(x, residual, weight, eps=1e-6, *, num_warps):
    """Allocate/launch the original arithmetic with an explicit warp count."""
    if type(num_warps) is not int or num_warps not in WARP_CANDIDATES:
        raise ValueError('num_warps must be one of 1, 2, 4, 8, 16')
    m, h = x.shape
    config = launch_config(h)
    output, residual_out = torch.empty_like(x), torch.empty_like(residual)
    with torch.cuda.device(x.device):
        _residual_rmsnorm_kernel[(m,)](
            x, residual, weight, output, residual_out,
            H=h, EPS=eps, BLOCK=config['block_size'],
            num_warps=num_warps, num_stages=1, enable_fp_fusion=False,
        )
    return output, residual_out


def launch_variant(x, residual, weight, eps=1e-6, *, num_warps):
    validate_inputs(x, residual, weight, eps)
    if torch.is_grad_enabled() and any(t.requires_grad for t in (x, residual, weight)):
        raise RuntimeError('This custom kernel supports forward inference only')
    return launch_variant_unchecked(x, residual, weight, eps, num_warps=num_warps)
