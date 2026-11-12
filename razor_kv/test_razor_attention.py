"""Correctness check for razor_attention (the eviction-aware attention used in RAZOR training).

uv run --with torch --with 'transformers>=4.56,<5.3' --with 'fire>=0.7' --with kvpress python razor_kv/test_razor_attention.py
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from razor_kv.train_razor import ST, razor_attention  # noqa: E402


def reference(q, k, v, scaling, keep):  # keep: (B, Hq, T, T) bool, True = query may attend key
    g = q.shape[1] // k.shape[1]
    k, v = k.repeat_interleave(g, 1), v.repeat_interleave(g, 1)
    s = (q @ k.transpose(-1, -2)) * scaling
    s = s.masked_fill(~keep, float("-inf"))
    return (s.softmax(-1) @ v).transpose(1, 2)


torch.manual_seed(0)
B, Hq, Hkv, T, D, W = 1, 4, 2, 12, 8, 3
q, k, v = (torch.randn(B, h, T, D, dtype=torch.float64) for h in (Hq, Hkv, Hkv))
sc = D ** -0.5
i = torch.arange(T)
causal = (i[None, :] <= i[:, None]).expand(B, Hq, T, T)
ST["window"] = W

# 1. gate off == plain causal attention
ST["on"] = False
out, _ = razor_attention(SimpleNamespace(), q, k, v, None, sc)
assert torch.allclose(out.double(), reference(q, k, v, sc, causal), atol=1e-5)

# 2. gate on, everything kept == plain causal attention
ST["on"] = True
mod = SimpleNamespace(_razor_z=torch.ones(B, Hkv, T))
out, _ = razor_attention(mod, q, k, v, None, sc)
assert torch.allclose(out.double(), reference(q, k, v, sc, causal), atol=1e-5)

# 3. evicted keys vanish only for queries at least W tokens later, and only for their own KV head group
z = (torch.rand(B, Hkv, T) > 0.5).double()
z[..., -1] = 1.0
mod = SimpleNamespace(_razor_z=z)
far = (i[:, None] - i[None, :]) >= W
keep = causal & ~(far & (z.repeat_interleave(Hq // Hkv, 1)[:, :, None, :] == 0))
out, _ = razor_attention(mod, q, k, v, None, sc)
assert torch.allclose(out.double(), reference(q, k, v, sc, keep), atol=1e-5)

# 4. straight-through: an evicted token (z_hard = 0) still receives gradient, so it can be brought back
p = torch.full((B, Hkv, T), 0.3, dtype=torch.float64, requires_grad=True)
mod = SimpleNamespace(_razor_z=(p > 0.5).double() + p - p.detach())
out, _ = razor_attention(mod, q, k, v, None, sc)
out.pow(2).sum().backward()
assert p.grad[..., : T - W].abs().sum() > 0, "no gradient reaches evicted far tokens"
assert p.grad[..., T - W + 1:].abs().sum() == 0, "tokens never far from any query must get no gradient"

print("razor_attention: all 4 checks passed")
