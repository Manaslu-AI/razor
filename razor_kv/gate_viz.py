"""What does the RAZOR gate keep? Run Qwen3-1.7B + the trained gate over wikitext documents and record, for every
layer x KV head, the fraction of tokens kept (logit > 0, i.e. tau=0, outside the 128-token window), keep rate by token
class, and per-token keep fraction for one example passage. Self-contained (no kvpress). Writes gate_viz.json."""
import json
import re
import sys

import torch
from datasets import load_dataset
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

GATE, N_DOCS, DOC_LEN, WIN = sys.argv[1], 16, 2048, 128
dev = "cuda" if torch.cuda.is_available() else "cpu"
print("device", dev, flush=True)
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B", dtype=torch.float16 if dev == "cuda" else torch.float32).to(dev).eval()
L = model.config.num_hidden_layers
w = load_file(GATE)
gate = [torch.nn.Sequential(torch.nn.Linear(2048, 512), torch.nn.GELU(), torch.nn.Linear(512, 8)) for _ in range(L)]
for i, g in enumerate(gate):
    g.load_state_dict({"0.weight": w[f"layers.{i}.0.weight"], "0.bias": w[f"layers.{i}.0.bias"],
                       "2.weight": w[f"layers.{i}.2.weight"], "2.bias": w[f"layers.{i}.2.bias"]})
    g.float().to(dev).eval()

logits = {}


def hook(i):
    def pre(module, args, kwargs):  # same input the gate saw in training: the attention block's hidden_states
        logits[i] = gate[i](kwargs["hidden_states"][0].float())  # (T, Hkv)
    return pre


for i, layer in enumerate(model.model.layers):
    layer.self_attn.register_forward_pre_hook(hook(i), with_kwargs=True)

# wikitext-103 test articles, concatenated into DOC_LEN-token documents
text = "".join(load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="test")["text"])
ids_all = tok(text, add_special_tokens=False).input_ids
FUNC = set("the a an of to in and or for on at by with from as is was are were be been that this it its which who".split())


def cls(t):
    s = t.strip()
    if "\n" in t:
        return "newline"
    if not s:
        return "space"
    if any(c.isdigit() for c in s):
        return "number"
    if all(not c.isalnum() for c in s):
        return "punct"
    if s.lower() in FUNC:
        return "function word"
    if t.startswith(" ") and s[0].isupper():
        return "capitalized word"
    if not t.startswith(" "):
        return "word piece"
    return "other word"


keep_lh = torch.zeros(L, 8)
cnt, cls_keep, cls_n = 0, {}, {}
example = None
for d in range(N_DOCS):
    ids = torch.tensor([ids_all[d * DOC_LEN:(d + 1) * DOC_LEN]], device=dev)
    with torch.no_grad():
        model(ids)
    k = torch.stack([(logits[i] > 0).float() for i in range(L)])[:, :-WIN].cpu()  # (L, T-WIN, H)
    keep_lh += k.mean(1)
    cnt += 1
    per_tok = k.mean((0, 2))  # fraction of (layer, head) pairs that keep each token
    toks = tok.convert_ids_to_tokens(ids[0, :-WIN].tolist())
    strs = [tok.convert_tokens_to_string([t]) for t in toks]
    for s, f in zip(strs, per_tok.tolist()):
        c = cls(s)
        cls_keep[c] = cls_keep.get(c, 0.0) + f
        cls_n[c] = cls_n.get(c, 0) + 1
    if d == 0:
        example = [[s, round(f, 3)] for s, f in zip(strs[200:520], per_tok[200:520].tolist())]
    print("doc", d, "keep", round(float(k.mean()), 3), flush=True)

out = dict(keep_layer_head=(keep_lh / cnt).tolist(), keep_by_class={c: cls_keep[c] / cls_n[c] for c in cls_n},
           count_by_class=cls_n, overall=float((keep_lh / cnt).mean()), example=example, n_docs=N_DOCS, doc_len=DOC_LEN)
json.dump(out, open("gate_viz.json", "w"))
print("DONE overall keep", round(out["overall"], 3), json.dumps({c: round(v, 3) for c, v in out["keep_by_class"].items()}), flush=True)
