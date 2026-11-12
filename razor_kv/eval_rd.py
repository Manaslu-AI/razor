"""One point on the KV cache's rate-distortion curve: held-out KL to the full model (nats/token, distortion) vs the
fraction of KV entries a gate keeps (rate), with hard gates at threshold(s) tau. Same exact masked attention as training.

Held-out docs = the "test" split train_razor.py sets aside (first 5 per Nemotron subset), so never trained on.
python eval_rd.py --gate_dir gates/razor --kvpress_src kvpress-src --taus 0,-1,1 --out rd_razor.json
"""
import json
import os
import sys

import torch
import torch.nn.functional as F
from fire import Fire
from transformers import AttentionInterface, AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main(gate_dir, kvpress_src, model_name="Qwen/Qwen3-1.7B", taus="0", out="rd.json", window=128,
         kl_positions=512, max_len=1024, min_tokens=750, max_tokens=1250, n_train_per_subset=200):
    sys.path.insert(0, kvpress_src)
    from kvpress.presses.kvzap_press import KVzapModel
    from kvzap.data import load_nemotron_dataset

    import razor_kv.train_razor as tr
    tr.ST["window"] = window
    AttentionInterface.register("razor", tr.razor_attention)
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float16, device_map="cuda",
                                                 attn_implementation="razor").eval()
    gate = KVzapModel.from_pretrained(gate_dir).cuda().float().eval()
    tau = [0.0]

    def pre_hook(module, args, kwargs):
        if tr.ST["on"]:
            s = gate.layers[module.layer_idx](kwargs["hidden_states"].float()).transpose(1, 2)  # (B, Hkv, T)
            module._razor_z = (s > tau[0]).float()

    for layer in model.model.layers:
        layer.self_attn.register_forward_pre_hook(pre_hook, with_kwargs=True)

    df = load_nemotron_dataset(tok, min_tokens, max_tokens, n_train_per_subset, 5)  # same call as training
    texts = df[df["split"] == "test"]["text"].tolist()
    res = []
    for t in [float(x) for x in (taus if isinstance(taus, (list, tuple)) else str(taus).split(","))]:  # fire parses "0,-1" as a tuple
        tau[0] = t
        kls, mem = [], []
        for text in texts:
            ids = tok(text, return_tensors="pt").input_ids[:, :max_len].cuda()
            with torch.no_grad():
                tr.ST["on"] = False
                p = F.log_softmax(model(ids).logits[0, -kl_positions:].float(), -1)
                tr.ST["on"] = True
                q = F.log_softmax(model(ids).logits[0, -kl_positions:].float(), -1)
                tr.ST["on"] = False
            kls.append(F.kl_div(q, p, log_target=True, reduction="none").sum(-1).mean().item())
            # memory actually held for the last query: recent window + gated keys older than the window
            T = ids.shape[1]
            z = torch.stack([layer.self_attn._razor_z[0] for layer in model.model.layers])  # (L, Hkv, T)
            mem.append(((z[..., : T - window].sum(-1) + min(window, T)) / T).mean().item())
        r = dict(tau=t, kl=sum(kls) / len(kls), kept=sum(mem) / len(mem), n_docs=len(texts))
        print("RD", gate_dir, r, flush=True)
        res.append(r)
    json.dump(dict(gate=gate_dir, points=res), open(out, "w"))


if __name__ == "__main__":
    Fire(main)
