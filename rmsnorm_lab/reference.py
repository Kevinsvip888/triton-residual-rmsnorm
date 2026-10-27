"""Explicit eager reference and validation for the version-1 operation contract."""
import math
import torch


def validate_inputs(x, residual, weight, eps=1e-6):
    """Validate metadata only; avoid tensor reductions or GPU synchronization."""
    if not all(isinstance(t, torch.Tensor) for t in (x, residual, weight)):
        raise TypeError('x, residual, and weight must be torch tensors')
    if x.ndim != 2 or x.shape[0] < 1 or x.shape[1] < 1:
        raise ValueError('x must have nonempty shape [M, H]')
    if residual.shape != x.shape:
        raise ValueError('residual must have the same shape as x')
    if weight.ndim != 1 or weight.shape[0] != x.shape[1]:
        raise ValueError('weight must have shape [H]')
    if x.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError('supported dtypes are float16 and bfloat16')
    if residual.dtype != x.dtype or weight.dtype != x.dtype:
        raise TypeError('all three tensors must have the same dtype')
    if not x.is_cuda or residual.device != x.device or weight.device != x.device:
        raise ValueError('all three tensors must be on the same CUDA device')
    if not all(t.is_contiguous() for t in (x, residual, weight)):
        raise ValueError('all three tensors must be contiguous')
    if isinstance(eps, bool) or not isinstance(eps, (int, float)):
        raise TypeError('eps must be a Python number')
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError('eps must be finite and positive')


def residual_rmsnorm_reference(x, residual, weight, eps=1e-6):
    """Return (normalized_output, rounded_residual_sum), without mutating inputs.

    Preconditions are checked separately by validate_inputs. Keep the tensor
    computation free of validation when comparing eager/compiled/Triton code.
    Inference callers should use torch.inference_mode and disable autocast.
    """
    residual_out = (x.float() + residual.float()).to(x.dtype)
    values = residual_out.float()
    mean_square = (values * values).mean(dim=-1, keepdim=True)
    inverse_rms = torch.rsqrt(mean_square + eps)
    output = ((values * inverse_rms) * weight.float()).to(x.dtype)
    return output, residual_out
