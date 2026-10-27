"""Frozen shape-specific launch policy; original arithmetic remains unchanged."""
from functools import lru_cache
import hashlib
import json
from pathlib import Path
from types import MappingProxyType

import torch
from .reference import validate_inputs
from .triton_kernel import launch_config
from .launch_variants import launch_variant_unchecked, WARP_CANDIDATES

ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = ROOT / 'outputs/day11/warp_tuning_v1/selected_policy.json'


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@lru_cache(maxsize=1)
def _load_policy():
    # Load before CUDA Graph capture. Never read files inside a JIT kernel.
    content = POLICY_PATH.read_bytes()
    policy = json.loads(content)
    if policy.get('study') != 'day11_selected_warp_policy_v1':
        raise RuntimeError('Unexpected selected policy')
    for field, relative in (
        ('kernel_source_sha256', 'rmsnorm_lab/triton_kernel.py'),
        ('variant_source_sha256', 'rmsnorm_lab/launch_variants.py'),
        ('contract_sha256', 'specs/contract_v1.json'),
    ):
        if policy.get(field) != _sha(ROOT / relative):
            raise RuntimeError('Frozen policy source mismatch: ' + relative)
    expected = {(d, m, h) for d in ('float16', 'bfloat16')
                for m in (1, 32, 2048) for h in (768, 1024, 2560, 4096)}
    choices = {}
    for key, item in policy['cases'].items():
        d, m, h, warps = item['dtype'], item['M'], item['H'], item['num_warps']
        if (type(m) is not int or type(h) is not int or type(warps) is not int
                or (d, m, h) not in expected or key != f'{d}_m{m}_h{h}'
                or warps not in WARP_CANDIDATES):
            raise RuntimeError('Invalid selected-policy entry: ' + key)
        choices[(d, m, h)] = warps
    if set(choices) != expected or len(policy['cases']) != 24:
        raise RuntimeError('Expected exactly 24 policy entries')
    return MappingProxyType(choices), hashlib.sha256(content).hexdigest()


def policy_sha256():
    """Hash of the policy actually loaded into this process."""
    return _load_policy()[1]


def selected_launch_config(dtype, m, h):
    config = dict(launch_config(h))
    label = {torch.float16: 'float16', torch.bfloat16: 'bfloat16'}.get(dtype)
    choices, _ = _load_policy()
    key = (label, m, h)
    config['matched_policy'] = key in choices
    config['num_warps'] = choices.get(key, config['num_warps'])
    return config


def residual_rmsnorm_tuned_unchecked(x, residual, weight, eps=1e-6):
    """Allocate and launch; metadata validation belongs outside timing."""
    config = selected_launch_config(x.dtype, *x.shape)
    return launch_variant_unchecked(x, residual, weight, eps,
                                    num_warps=config['num_warps'])


def residual_rmsnorm_tuned(x, residual, weight, eps=1e-6):
    validate_inputs(x, residual, weight, eps)
    if torch.is_grad_enabled() and any(t.requires_grad for t in (x, residual, weight)):
        raise RuntimeError('This custom kernel supports forward inference only')
    return residual_rmsnorm_tuned_unchecked(x, residual, weight, eps)
