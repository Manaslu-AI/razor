"""RAZOR gate training: end-to-end MDL distillation.

KVzap regresses a hand-designed importance proxy (KVzip+). RAZOR instead trains the same gate
(KVzapModel MLP) so that the *evicted* model matches the full model's next-token distribution,
while paying an MDL price for every token it keeps:

    L = KL(p_full || p_evicted) + lambda * mean(p_keep)        lambda <- dual ascent to target keep rate

Eviction is hard and binary in the forward pass (z = 1[p > 0.5]), straight-through in the backward
pass (the RAZOR thesis). Like DMS/KVzap inference, tokens inside a recent window are never evicted.
"""
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from fire import Fire
from transformers import AttentionInterface, AutoModelForCausalLM, AutoTokenizer

from kvpress.presses.kvzap_press import KVzapConfig, KVzapModel

ST = {"on": False, "window": 128, "p": {}, "n_ctx": None}


def razor_attention(module, query, key, value, attention_mask, scaling, dropout=0.0, **kw):
    """Exact attention in fp32 where far keys are multiplied by the (STE) gate z before normalisation."""
    _, hq, t, _ = query.shape
    g = hq // key.shape[1]
    k, v = key.repeat_interleave(g, 1).float(), value.repeat_interleave(g, 1).float()
    s = (query.float() @ k.transpose(-1, -2)) * scaling
    i = torch.arange(t, device=s.device)
    s = s.masked_fill(i[None, :] > i[:, None], float("-inf"))
    e = (s - s.amax(-1, keepdim=True)).exp()
    if ST["on"]:
        z = module._razor_z.to(e.dtype).repeat_interleave(g, 1)[:, :, None, :]  # (B, Hq, 1, T)
        far = (i[:, None] - i[None, :]) >= ST["window"]
        e = torch.where(far, e * z, e)
    a = e / e.sum(-1, keepdim=True)
    return (a @ v).to(query.dtype).transpose(1, 2).contiguous(), None


def main(
    model_name="Qwen/Qwen3-1.7B",
    output_dir="gates/razor",
    kvpress_src="kvpress-src",
    n_train_per_subset=200,
    min_tokens=750,
    max_tokens=1250,
    max_len=1024,
    kl_positions=512,
    window=128,
    target_keep=0.3,
    lr=1e-3,
    eta_lambda=0.5,
    hidden_dim=512,
    epochs=1,
    use_mdl=True,
    hard=True,
    seed=0,
    objective="lm",  # "lm": KL on the text's own continuation; "recon": KL on re-stating the whole context
    kl_top=1.0,  # <1: average KL over only the hardest fraction of positions
    data="nemotron",  # "openr1": DeepSeek-R1 math reasoning traces, to teach the gate what to forget while thinking
):
    torch.manual_seed(seed)
    sys.path.insert(0, kvpress_src)
    from kvzap.data import load_nemotron_dataset  # same data source as the KVzap baseline

    ST["window"] = window
    AttentionInterface.register("razor", razor_attention)
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float16, device_map="cuda", attn_implementation="razor")
    model.requires_grad_(False)
    # recompute layers in backward instead of storing 28 fp32 attention maps (T4 has 15 GB)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()  # checkpointing is only active in train mode; Qwen3 has no dropout
    cfg = model.config
    gate = KVzapModel(KVzapConfig(input_dim=cfg.hidden_size, hidden_dim=hidden_dim,
                                  output_dim=cfg.num_key_value_heads, n_modules=cfg.num_hidden_layers)).cuda().float()
    for layer in gate.layers:  # start mostly open (p ~ 0.82), as in the original RAZOR gate
        torch.nn.init.constant_(layer[-1].bias, 1.5)

    def pre_hook(module, args, kwargs):
        if ST["on"]:
            logit = gate.layers[module.layer_idx](kwargs["hidden_states"].float())  # (B, T, Hkv)
            p = torch.sigmoid(logit).transpose(1, 2)  # (B, Hkv, T)
            z = ((p > 0.5).float() + p - p.detach()) if hard else p
            n = ST["n_ctx"]
            if n:  # recon: only the context is compressible; the probe tokens after it are never evicted
                z, p = torch.cat([z[..., :n], torch.ones_like(z[..., n:])], -1), p[..., :n]
            module._razor_z = z
            ST["p"][module.layer_idx] = p  # dict: checkpoint recompute overwrites instead of duplicating

    for layer in model.model.layers:
        layer.self_attn.register_forward_pre_hook(pre_hook, with_kwargs=True)

    if data == "openr1":
        from datasets import load_dataset
        texts = []
        for r in load_dataset("open-r1/OpenR1-Math-220k", split="train", streaming=True):
            t = f"<|im_start|>user\n{r['problem']}<|im_end|>\n<|im_start|>assistant\n{r['generations'][0]}"
            if len(tok(t).input_ids) >= min_tokens:
                texts.append(t)
            if len(texts) >= 5 * n_train_per_subset:
                break
    else:
        df = load_nemotron_dataset(tok, min_tokens, max_tokens, n_train_per_subset, 5)
        texts = df[df["split"] == "train"]["text"].sample(frac=1.0, random_state=seed).tolist()
    probe = tok("\n\nRepeat the previous context exactly.\n", return_tensors="pt", add_special_tokens=False).input_ids
    opt = torch.optim.AdamW(gate.parameters(), lr=lr, weight_decay=0.0)
    lam, step, t0 = 0.0, 0, time.time()
    total = epochs * len(texts)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total)
    log = []

    for _ in range(epochs):
        for text in texts:
            ids = tok(text, return_tensors="pt").input_ids[:, :max_len]
            sl = slice(-kl_positions, None)
            if objective == "recon":  # query-agnostic: the compressed cache must still let the model restate everything
                n = ids.shape[1]
                ids = torch.cat([ids, probe, ids], 1)
                sl, ST["n_ctx"] = slice(-n - 1, -1), n
            ids = ids.cuda()
            with torch.no_grad():
                ST["on"] = False
                t_logp = F.log_softmax(model(ids).logits[0, sl].float(), -1)
            ST["on"], ST["p"] = True, {}
            s_logp = F.log_softmax(model(ids).logits[0, sl].float(), -1)
            kl_pos = F.kl_div(s_logp, t_logp, log_target=True, reduction="none").sum(-1)  # (P,)
            kl = kl_pos.topk(max(1, int(kl_top * kl_pos.numel()))).values.mean()
            p_all = torch.stack([p[0] for p in ST["p"].values()])  # (L, Hkv, T)
            keep = p_all.mean()
            hard_keep = (p_all > 0.5).float().mean().item()
            loss = kl + (lam * keep if use_mdl else 0.0)
            opt.zero_grad(set_to_none=True)
            loss.backward()  # gate must stay on here: checkpointed layers are recomputed in backward
            ST["on"] = False
            torch.nn.utils.clip_grad_norm_(gate.parameters(), 1.0)
            opt.step()
            sched.step()
            if use_mdl:  # dual ascent: raise the price of memory while we keep more than the budget
                lam = max(0.0, lam + eta_lambda * (hard_keep - target_keep))
            step += 1
            if step % 25 == 0 or step == total:
                rec = dict(step=step, kl=round(kl.item(), 5), keep_soft=round(keep.item(), 4),
                           keep_hard=round(hard_keep, 4), lam=round(lam, 4), sec=round(time.time() - t0))
                log.append(rec)
                print(rec, flush=True)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    gate.half().save_pretrained(out)
    (out / "train_log.txt").write_text("\n".join(map(str, log)))
    print(f"saved gate to {out}")


if __name__ == "__main__":
    Fire(main)
