"""How long a context fits on ONE GPU, full cache vs RAZOR with real deletion (ragged.py)?

For each length: passkey haystack -> prefill -> answer. Reports peak GPU memory, KV bytes, prefill seconds,
decode tokens/s and whether the passkey came back. A failed allocation is recorded as OOM.
python bench_memory.py --lengths 16000,32000,64000,100000,128000 --gate $RAZOR_GATE
"""
import gc
import json
import random
import re
import time

import torch
import transformers.integrations.sdpa_attention as _sdpa
from fire import Fire
from kvpress import KVPressTextGenerationPipeline  # noqa: F401  (registers the pipeline)
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, DynamicCache, pipeline

from razor_kv.press import LocalGatePress
from razor_kv.ragged import RaggedCache, RaggedDMSPress, enable

_sdpa.use_gqa_in_sdpa = lambda *a, **k: False  # T4: enable_gqa -> math kernel -> full T x T scores
FILLER = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. "


def haystack(tok, n, key, depth=0.5):
    reps = n // len(tok(FILLER).input_ids)
    needle = f"The pass key is {key}. Remember it. {key} is the pass key. "
    parts = [FILLER] * reps
    parts.insert(int(depth * reps), needle)
    return ("There is an important info hidden inside a lot of irrelevant text. Find it and memorize it. "
            "I will quiz you about the important information there.\n\n" + "".join(parts))  # InfiniteBench passkey prompt


def decode_speed(model, cache, steps=64):
    """Greedy-decode `steps` tokens on top of the cached context (no eviction during the timing), tokens/s."""
    n, tok = cache.get_seq_length(), torch.tensor([[0]], device=model.device)
    with torch.no_grad():
        for i in range(steps + 8):  # 8 warm-up steps
            if i == 8:
                torch.cuda.synchronize()
                t0 = time.time()
            tok = model(input_ids=tok, past_key_values=cache, position_ids=torch.tensor([[n + i]], device=model.device)).logits[:, -1:].argmax(-1)
    torch.cuda.synchronize()
    return round(steps / (time.time() - t0), 1)


def main(lengths="16000,32000,64000,100000,128000", gate="", threshold=0.0, model_name="Qwen/Qwen3-1.7B",
         methods="razor,full", out="bench_memory.json", seed=0, chunk=0):
    tok = AutoTokenizer.from_pretrained(model_name)
    cfg = AutoConfig.from_pretrained(model_name)
    cfg.rope_parameters = {**cfg.rope_parameters, "rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 32768}
    cfg.max_position_embeddings = 131072
    model = AutoModelForCausalLM.from_pretrained(model_name, config=cfg, dtype=torch.float16, device_map="cuda")
    enable(model)  # ragged attention; falls back to plain SDPA for a normal DynamicCache
    pipe = pipeline("kv-press-text-generation", model=model, tokenizer=tok)
    weights = torch.cuda.memory_allocated()
    rows = []
    for n in map(int, re.findall(r"\d+", str(lengths))):  # Fire turns "a,b" into a tuple
        key = random.Random(seed + n).randint(10000, 99999)
        ctx = haystack(tok, n, key)
        for m in re.findall(r"[a-z]+", str(methods)):
            press = RaggedDMSPress(LocalGatePress(path=gate), threshold=threshold, prefill_chunk=chunk) if m == "razor" else None
            cache = RaggedCache() if m == "razor" else DynamicCache()
            gc.collect(), torch.cuda.empty_cache(), torch.cuda.reset_peak_memory_stats()
            row = dict(model=model_name, method=m, chunk=chunk if m == "razor" else 0, tokens=len(tok(ctx).input_ids))
            try:
                t0 = time.time()
                with torch.no_grad():
                    ans = pipe(ctx, question="What is the pass key?", answer_prefix="The pass key is", press=press, cache=cache, max_new_tokens=16)["answer"]
                torch.cuda.synchronize()
                row["decode_tok_s"] = decode_speed(model, cache)
                kv = cache.nbytes() if m == "razor" else sum(t.numel() * t.element_size() for l in cache.layers for t in (l.keys, l.values))
                row.update(ok=str(key) in ans, answer=ans[:60], seconds=round(time.time() - t0, 1),
                           peak_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2), kv_gb=round(kv / 1e9, 3),
                           weights_gb=round(weights / 1e9, 2), evicted=round(press.compression_ratio, 3) if press else 0.0)
            except torch.OutOfMemoryError:
                row.update(ok=False, answer="OOM", peak_gb=None)
            del cache
            rows.append(row)
            print("MEM", json.dumps(row), flush=True)
            json.dump(rows, open(out, "w"), indent=1)


if __name__ == "__main__":
    Fire(main)
