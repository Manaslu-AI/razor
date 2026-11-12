"""Compress a thinking model's KV cache *while it reasons* (DMSPress decoding=True), score on MATH-500.

python eval_reason.py --press razor --threshold 0 --n 60 --out reason
press: no_press | razor | kvzap_ours (gate dirs from env RAZOR_GATE / KVZAP_GATE)
"""
import json
import os
import re
import sys
import time
from contextlib import nullcontext

import torch
from datasets import load_dataset
from fire import Fire
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def boxed(text):
    """Content of the last \\boxed{...}, braces balanced."""
    i = text.rfind("\\boxed{")
    if i < 0:
        return None
    i, depth, out = i + 7, 1, []
    while i < len(text):
        c = text[i]
        depth += (c == "{") - (c == "}")
        if depth == 0:
            return "".join(out)
        out.append(c)
        i += 1
    return None


def norm(s):
    s = re.sub(r"\\(left|right|displaystyle|!|,|;|\s)", "", s or "")
    s = s.replace("dfrac", "frac").replace("tfrac", "frac").replace("^{\\circ}", "").replace("^\\circ", "")
    s = re.sub(r"\\text\{\s*([^}]*)\}", r"\1", s)
    return s.replace(" ", "").replace("$", "").rstrip(".")


def main(press="no_press", threshold=0.0, n=60, max_new_tokens=6000, model_name="Qwen/Qwen3-1.7B",
         out="reason", seed=0):
    from kvpress import DMSPress  # imported here so boxed/norm stay usable without kvpress
    from razor_kv.press import LocalGatePress

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float16, device_map="cuda")
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test").filter(lambda r: r["level"] >= 4)
    ds = ds.shuffle(seed=0).select(range(min(n, len(ds))))  # same problems for every method
    path = {"razor": os.environ.get("RAZOR_GATE"), "kvzap_ours": os.environ.get("KVZAP_GATE"),
            "razor_math": os.environ.get("GATE_RAZOR_MATH")}.get(press)
    pr = DMSPress(LocalGatePress(path=path), threshold=threshold, decoding=True) if path else None
    rows, t0 = [], time.time()
    os.makedirs(out, exist_ok=True)
    name = f"{press}_{threshold:+.1f}" if pr else press
    for k, r in enumerate(ds):
        msgs = [{"role": "user", "content": r["problem"] + "\nPut your final answer within \\boxed{}."}]
        ids = tok.apply_chat_template(msgs, return_tensors="pt", return_dict=True, add_generation_prompt=True).input_ids.cuda()
        torch.manual_seed(seed + k)
        with pr(model) if pr else nullcontext():
            o = model.generate(ids, do_sample=True, temperature=0.6, top_p=0.95, top_k=20, max_new_tokens=max_new_tokens)
        text = tok.decode(o[0, ids.shape[1]:], skip_special_tokens=True)
        pred = boxed(text)
        rows.append(dict(i=k, level=r["level"], answer=r["answer"], pred=pred, correct=norm(pred) == norm(r["answer"]),
                         gen_tokens=o.shape[1] - ids.shape[1], evicted=float(pr.compression_ratio) if pr else 0.0))
        json.dump(dict(press=name, partial=True, rows=rows), open(f"{out}/{name}.json", "w"))  # survive a 12h kill
        print(k, rows[-1]["correct"], rows[-1]["gen_tokens"], round(rows[-1]["evicted"], 3), f"{time.time()-t0:.0f}s", flush=True)
    acc = sum(x["correct"] for x in rows) / len(rows)
    ev = sum(x["evicted"] for x in rows) / len(rows)
    toks = sum(x["gen_tokens"] for x in rows) / len(rows)
    json.dump(dict(press=name, acc=acc, evicted=ev, gen_tokens=toks, rows=rows), open(f"{out}/{name}.json", "w"))
    print(f"REASON {name} acc={acc:.3f} evicted={ev:.3f} gen_tokens={toks:.0f}", flush=True)


if __name__ == "__main__":
    if sys.argv[1:2] == ["--selftest"]:
        assert boxed(r"so \boxed{\frac{1}{2}} done") == r"\frac{1}{2}"
        assert norm(r"\dfrac{1}{2}") == norm(r"\frac{1}{2}") and norm(r"10^\circ") == "10" and boxed("none") is None
        print("selftest ok")
    else:
        Fire(main)
