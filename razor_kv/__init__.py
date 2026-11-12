"""RAZOR: a learned KV-cache forget gate. Evicted KV entries are deleted from GPU memory.

    import razor_kv
    pipe = razor_kv.pipeline("Qwen/Qwen3-4B")
    pipe(long_document, question="...")["answer"]
"""
from razor_kv.press import LocalGatePress
from razor_kv.ragged import RaggedCache, RaggedDMSPress, enable

GATES = {
    "Qwen/Qwen3-1.7B": "Manalsu/razor-gate-Qwen3-1.7B",
    "Qwen/Qwen3-4B": "Manalsu/razor-gate-Qwen3-4B",
}


def press(model_name: str, threshold: float = 0.0, prefill_chunk: int = 4096, gate: str | None = None) -> RaggedDMSPress:
    """The RAZOR press for a base model. `gate` overrides the HF gate (repo id or local dir).
    Higher threshold evicts more. prefill_chunk bounds activation memory on long inputs (0 = one-shot prefill)."""
    if gate is None and model_name not in GATES:
        raise ValueError(f"No released gate for {model_name}. Released: {', '.join(GATES)}. Pass gate= to use your own.")
    return RaggedDMSPress(LocalGatePress(path=gate or GATES[model_name]), threshold=threshold, prefill_chunk=prefill_chunk)


def pipeline(model_name: str, threshold: float = 0.0, prefill_chunk: int = 4096, gate: str | None = None, **pipeline_kwargs):
    """A kvpress text-generation pipeline with RAZOR on. Call it as pipe(context, question=..., max_new_tokens=...).
    Batch size 1. Extra kwargs go to transformers.pipeline (default dtype float16, device_map auto)."""
    from kvpress import KVPressTextGenerationPipeline  # noqa: F401, registers "kv-press-text-generation"
    from transformers import pipeline as hf_pipeline

    p = press(model_name, threshold, prefill_chunk, gate)
    pipe = hf_pipeline("kv-press-text-generation", model=model_name,
                       **{"dtype": "float16", "device_map": "auto", **pipeline_kwargs})
    enable(pipe.model)

    def run(context: str, **kwargs):
        return pipe(context, press=p, cache=RaggedCache(), **kwargs)  # fresh cache per call

    run.press, run.pipe = p, pipe
    return run


__all__ = ["GATES", "LocalGatePress", "RaggedCache", "RaggedDMSPress", "enable", "pipeline", "press"]
