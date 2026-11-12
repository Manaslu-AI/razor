"""Real KV memory freeing for threshold (DMS) eviction: every KV head keeps only its own surviving tokens.

kvpress's DMSPress only *masks* evicted keys (the cache stays seq_len x heads). Here evicted keys/values are
physically deleted, layer by layer as soon as that layer has been prefilled, so cache memory is the sum over
heads of kept tokens and peak memory is that plus one uncompressed layer. Attention runs per KV head over
its own ragged cache (same math as masking; see __main__ for the equivalence check). Batch size 1.

    from razor_kv.ragged import RaggedCache, RaggedDMSPress, enable
    enable(model)                                   # registers + selects the ragged attention
    press = RaggedDMSPress(LocalGatePress(path=gate), threshold=0.0)
    pipe(context, question=q, press=press, cache=RaggedCache())
"""
from contextlib import contextmanager
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from kvpress import DMSPress
from kvpress.pipeline import KVPressTextGenerationPipeline
from kvpress.presses.base_press import is_prefilling
from transformers import AttentionInterface, DynamicCache
from transformers.cache_utils import DynamicLayer
from transformers.integrations.sdpa_attention import sdpa_attention_forward


class RaggedLayer(DynamicLayer):
    """Per KV head h: k[h], v[h] [kept_h, head_dim] and pos[h] (absolute positions, ascending) are views into
    preallocated buffers, so a decode step writes in place instead of copying the head. n: tokens seen."""

    def lazy_initialization(self, key_states, value_states):
        super().lazy_initialization(key_states, value_states)
        H, D = key_states.shape[1], key_states.shape[-1]
        self._k = [key_states.new_empty(0, D) for _ in range(H)]
        self._v = [value_states.new_empty(0, D) for _ in range(H)]
        self._p = [torch.empty(0, dtype=torch.long, device=key_states.device) for _ in range(H)]
        self.L = [0] * H
        self.n = 0
        self._views()

    def _views(self):
        self.k = [b[:L] for b, L in zip(self._k, self.L)]
        self.v = [b[:L] for b, L in zip(self._v, self.L)]
        self.pos = [b[:L] for b, L in zip(self._p, self.L)]

    def _resize(self, h, cap):
        L = self.L[h]
        for bufs in (self._k, self._v, self._p):
            b = bufs[h].new_empty(cap, *bufs[h].shape[1:])
            b[:L] = bufs[h][:L]
            bufs[h] = b

    def update(self, key_states, value_states, cache_kwargs=None):
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)
        assert key_states.shape[0] == 1, "RaggedLayer supports batch size 1"
        q = key_states.shape[2]
        new = torch.arange(self.n, self.n + q, device=key_states.device)
        for h in range(len(self.L)):
            L = self.L[h]
            if L + q > self._k[h].shape[0]:  # grow 12.5% (+128): amortised O(1) appends, small slack
                self._resize(h, L + q + max(128, (L + q) // 8))
            self._k[h][L:L + q], self._v[h][L:L + q], self._p[h][L:L + q] = key_states[0, h], value_states[0, h], new
            self.L[h] = L + q
        self.n += q
        self._views()
        return self, self  # the ragged attention below reads the heads from the layer itself

    def get_seq_length(self):
        return self.n if self.is_initialized else 0

    def evict(self, drop, start):
        """drop: [H, m] bool for absolute positions start..start+m-1. Only the tail from `start` moves."""
        for h in range(len(self.L)):
            if not drop[h].any():
                continue
            L, p = self.L[h], self._p[h]
            i0 = int(torch.searchsorted(p[:L], torch.tensor(start, device=p.device)))
            keep = ~torch.isin(p[i0:L], start + drop[h].nonzero().squeeze(1))
            m = int(keep.sum())
            for b in (self._k[h], self._v[h], p):
                b[i0:i0 + m] = b[i0:L][keep]
            self.L[h] = i0 + m
            if self._k[h].shape[0] > 2 * self.L[h] + 512:  # hand memory back after a big eviction
                self._resize(h, self.L[h] + max(128, self.L[h] // 8))
        self._views()

    def crop(self, max_length):
        for h in range(len(self.L)):
            self.L[h] = int(torch.searchsorted(self._p[h][:self.L[h]], torch.tensor(max_length, device=self._p[h].device)))
        self.n = min(self.n, max_length)
        self._views()

    def kept(self):
        return sum(self.L)

    def capacity_bytes(self):
        return sum(b.numel() * b.element_size() for b in self._k + self._v)


class RaggedCache(DynamicCache):
    def __init__(self):
        super().__init__()
        self.layer_class_to_replicate = RaggedLayer

    def nbytes(self):  # bytes allocated for K/V, slack included (what the GPU actually holds)
        return sum(layer.capacity_bytes() for layer in self.layers)


def ragged_attention(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kwargs):
    if not isinstance(key, RaggedLayer):
        return sdpa_attention_forward(module, query, key, value, attention_mask, scaling=scaling, dropout=dropout, **kwargs)
    layer, H, q = key, len(key.k), query.shape[2]
    g = query.shape[1] // H
    outs = []
    for h in range(H):
        K, V = layer.k[h], layer.v[h]
        L, D = K.shape
        # the q new tokens are the last q entries of every head (eviction happens after attention)
        mask = None if L == q else (torch.arange(L, device=K.device)[None] <= L - q + torch.arange(q, device=K.device)[:, None])
        K, V = K.expand(1, g, L, D), V.expand(1, g, L, D)
        if q > 1:  # stride-0 heads can push SDPA to the math kernel (L x L scores); a g-fold copy of one head is cheap
            K, V = K.contiguous(), V.contiguous()
        outs.append(F.scaled_dot_product_attention(
            query[:, h * g:(h + 1) * g], K, V,
            attn_mask=mask, is_causal=mask is None and q > 1, scale=scaling))
    return torch.cat(outs, 1).transpose(1, 2).contiguous(), None


@dataclass
class RaggedDMSPress(DMSPress):
    """DMSPress (same threshold + sliding window logic) that deletes instead of masking. Needs RaggedCache.

    prefill_chunk > 0 prefills the context in chunks of that many tokens: activations scale with the chunk, not the
    context, and later chunks only attend to what earlier chunks kept (streaming, as at decode time)."""
    prefill_chunk: int = 0
    _chunking = False

    @contextmanager
    def __call__(self, model):
        with super().__call__(model):
            inner, c = model.model, self.prefill_chunk
            fwd = inner.forward

            def chunked(input_ids=None, **kw):
                if not c or input_ids is None or input_ids.shape[1] <= c:
                    return fwd(input_ids=input_ids, **kw)
                for i in range(0, input_ids.shape[1], c):
                    self._chunking = i > 0
                    out = fwd(input_ids=input_ids[:, i:i + c], **kw)
                self._chunking = False
                return out

            inner.forward = chunked
            try:
                yield
            finally:
                del inner.forward

    def forward_hook(self, module, input, kwargs, output):
        hs, li = kwargs["hidden_states"], module.layer_idx
        layer = kwargs["past_key_values"].layers[li]
        prefilling = is_prefilling(kwargs["cache_position"], hs.shape[1])
        if prefilling and li == 0:
            self.scores_buffer.clear()
            self.compression_ratios.clear()
        if not prefilling and not self.decoding and not self._chunking:
            return output
        s = self.press.score(module, hs, None, None, None, kwargs)  # gate scorers only need hidden states
        buf = s if prefilling else torch.cat([self.scores_buffer[li], s], -1)
        n_ev = buf.shape[-1] - self.sliding_window_size
        if n_ev > 0:
            layer.evict(buf[0, :, :n_ev] < self.threshold, layer.n - n_ev - self.sliding_window_size)
            buf = buf[..., n_ev:]
        self.scores_buffer[li] = buf
        self.compression_ratios[li] = 1 - layer.kept() / (len(layer.L) * layer.n)
        return output


def enable(model):
    AttentionInterface.register("razor_ragged", ragged_attention)
    model.set_attn_implementation("razor_ragged")


_orig_remove = KVPressTextGenerationPipeline._remove_answer_from_cache


def _remove_answer_from_cache(self, cache, cache_seq_lengths, layer_states=None):
    if not isinstance(cache, RaggedCache):
        return _orig_remove(self, cache, cache_seq_lengths, layer_states)
    for layer, n in zip(cache.layers, cache_seq_lengths):
        layer.crop(n)


KVPressTextGenerationPipeline._remove_answer_from_cache = _remove_answer_from_cache
_orig_sanitize = KVPressTextGenerationPipeline._sanitize_parameters


def _sanitize_parameters(self, *args, press=None, cache=None, **kwargs):
    if isinstance(press, RaggedDMSPress) and cache is None:  # a RaggedDMSPress brings its own cache + attention
        enable(self.model)
        cache = RaggedCache()
    return _orig_sanitize(self, *args, press=press, cache=cache, **kwargs)


KVPressTextGenerationPipeline._sanitize_parameters = _sanitize_parameters


if __name__ == "__main__":  # ragged deletion must give the same logits as kvpress masking
    from dataclasses import dataclass

    from kvpress.presses.scorer_press import ScorerPress
    from transformers import Qwen3Config, Qwen3ForCausalLM

    @dataclass
    class FakeGate(ScorerPress):  # deterministic per-head scores from hidden states
        def score(self, module, hidden_states, keys, values, attentions, kwargs):
            w = torch.Generator().manual_seed(module.layer_idx)
            W = torch.randn(hidden_states.shape[-1], 2, generator=w)
            return (hidden_states.float() @ W).transpose(1, 2)

    cfg = Qwen3Config(vocab_size=100, hidden_size=64, intermediate_size=128, num_hidden_layers=3, num_attention_heads=4,
                      num_key_value_heads=2, head_dim=16, max_position_embeddings=512)
    torch.manual_seed(0)
    model = Qwen3ForCausalLM(cfg).eval()
    ctx, qst = torch.randint(0, 100, (1, 200)), torch.randint(0, 100, (1, 7))

    def run(ragged, decoding):
        model.set_attn_implementation("sdpa")
        if ragged:
            enable(model)
        P = RaggedDMSPress if ragged else DMSPress
        press, cache = P(FakeGate(), threshold=0.0, sliding_window_size=16, decoding=decoding), RaggedCache() if ragged else DynamicCache()
        logits = []
        with torch.no_grad():
            with press(model):
                model.model(input_ids=ctx, past_key_values=cache)
                ratio = press.compression_ratio
                out = model(input_ids=qst, past_key_values=cache, position_ids=torch.arange(200, 207)[None])
                logits.append(out.logits)
                for t in range(40):  # long enough for decode-time eviction to kick in
                    out = model(input_ids=out.logits[:, -1:].argmax(-1), past_key_values=cache, position_ids=torch.tensor([[207 + t]]))
                    logits.append(out.logits)
        return torch.cat(logits, 1), ratio, cache

    for decoding in (False, True):
        a, ra, _ = run(False, decoding)
        b, rb, cache = run(True, decoding)
        err = (a - b).abs().max().item()
        print(f"decoding={decoding}: evicted masked={ra:.3f} ragged={rb:.3f} max|dlogit|={err:.2e} "
              f"kept={sum(l.kept() for l in cache.layers)} of {3 * 2 * cache.layers[0].n} head-tokens")
        assert err < 1e-4 and abs(ra - rb) < 1e-6 and 0.2 < rb < 0.9
    # chunked prefill: no eviction -> identical to one-shot; with eviction -> similar ratio, bounded change
    for t, tol in ((-1e9, 1e-4), (0.0, None)):
        out = {}
        for chunk in (0, 64):
            enable(model)
            press, cache = RaggedDMSPress(FakeGate(), threshold=t, sliding_window_size=16, prefill_chunk=chunk), RaggedCache()
            with torch.no_grad(), press(model):
                model.model(input_ids=ctx, past_key_values=cache)
                r = press.compression_ratio
            with torch.no_grad():
                out[chunk] = (model(input_ids=qst, past_key_values=cache, position_ids=torch.arange(200, 207)[None]).logits, r)
        err = (out[0][0] - out[64][0]).abs().max().item()
        print(f"chunked t={t}: evicted one-shot={out[0][1]:.3f} chunked={out[64][1]:.3f} max|dlogit|={err:.2e}")
        assert abs(out[0][1] - out[64][1]) < 0.15 and (tol is None or err < tol)
        assert "forward" not in model.model.__dict__  # chunking wrapper removed on exit
    # big eviction must hand memory back; tail-only compaction must keep order
    c = RaggedCache()
    c.update(torch.randn(1, 2, 4000, 16), torch.randn(1, 2, 4000, 16), 0)
    full = c.layers[0].capacity_bytes()
    drop = torch.rand(2, 3900, generator=torch.Generator().manual_seed(0)) < 0.9
    c.layers[0].evict(drop, 0)
    L0 = c.layers[0]
    assert L0.kept() == int((~drop).sum()) + 2 * 100 and L0.capacity_bytes() < 0.2 * full
    assert all(bool((p[1:] > p[:-1]).all()) for p in L0.pos)
    c = RaggedCache()
    c.update(torch.randn(1, 2, 10, 16), torch.randn(1, 2, 10, 16), 0)
    c.layers[0].crop(6)
    assert c.get_seq_length() == 6 and c.layers[0].k[1].shape[0] == 6
    print("ragged selftest ok")
