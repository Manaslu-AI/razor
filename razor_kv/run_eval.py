"""Run the official kvpress evaluation with locally trained gates registered.

Usage (from kvpress-src/evaluation):
  RAZOR_GATE=Manaslu-AI/razor-gate-Qwen3-1.7B python /path/to/razor/razor_kv/run_eval.py --press_name razor_dms --threshold -2 ...
Gate dirs come from env RAZOR_GATE / KVZAP_GATE.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.getcwd())

import transformers.integrations.sdpa_attention as _sdpa  # noqa: E402
from fire import Fire  # noqa: E402

# T4 has no flash kernel: SDPA with enable_gqa falls back to the math kernel and materialises the full
# T x T score matrix (15.8 GB at 16k). Expanding KV heads instead keeps the memory-efficient kernel.
_sdpa.use_gqa_in_sdpa = lambda *a, **k: False

if os.environ.get("RAZOR_YARN"):  # Qwen3 is native to 32k; YaRN xF extends it (Qwen's recommended long-context setup)
    from transformers import AutoConfig

    _orig_cfg = AutoConfig.from_pretrained

    def _yarn_cfg(*a, **k):
        out, f = _orig_cfg(*a, **k), float(os.environ["RAZOR_YARN"])
        c = out[0] if isinstance(out, tuple) else out  # (config, unused_kwargs) when return_unused_kwargs=True
        r = {"rope_type": "yarn", "factor": f, "original_max_position_embeddings": 32768}
        if getattr(c, "rope_parameters", None) is not None:  # transformers 5.x
            c.rope_parameters = {**c.rope_parameters, **r}
        else:
            c.rope_scaling = r
        c.max_position_embeddings = int(32768 * f)
        print("YaRN config:", getattr(c, "rope_parameters", None) or c.rope_scaling, flush=True)
        return out

    AutoConfig.from_pretrained = staticmethod(_yarn_cfg)
from kvpress import AdaKVPress, DMSPress  # noqa: E402

import evaluate  # noqa: E402
from evaluate_registry import PRESS_REGISTRY  # noqa: E402
from razor_kv.press import LocalGatePress  # noqa: E402
from razor_kv.ragged import RaggedDMSPress  # noqa: E402

gates = {"razor": os.environ.get("RAZOR_GATE"), "kvzap_ours": os.environ.get("KVZAP_GATE")}
gates.update({k[5:].lower(): v for k, v in os.environ.items() if k.startswith("GATE_")})  # GATE_RAZOR_SOFT=/path
for name, path in gates.items():
    if path:
        PRESS_REGISTRY[f"{name}_head"] = LocalGatePress(path=path)  # fixed compression ratio
        PRESS_REGISTRY[f"{name}_ada"] = AdaKVPress(LocalGatePress(path=path))  # fixed ratio, budget shared across heads
        PRESS_REGISTRY[f"{name}_dms"] = DMSPress(press=LocalGatePress(path=path))  # threshold + window
        PRESS_REGISTRY[f"{name}_ragged"] = RaggedDMSPress(press=LocalGatePress(path=path), prefill_chunk=int(os.environ.get("RAZOR_CHUNK", 0)))  # evicted KV really freed

if __name__ == "__main__":
    Fire(evaluate.CliEntryPoint)
