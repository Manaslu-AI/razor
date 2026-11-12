# RAZOR: a learned forget gate for the KV cache

A small gate (one MLP per layer, one score per KV head) decides, when each token is written, whether the model will
need it again. Tokens it rejects are deleted from GPU memory. The base model stays frozen, and the gate trains in
39 minutes on one 16 GB T4.

Training uses hard keep/drop decisions with straight-through gradients, KL divergence to the uncompressed model's
next-token distribution, and a price on every kept token. Dual ascent sets the price to meet a memory budget, so there
is no penalty weight to tune. At inference a token is kept if its score is above a threshold (DMS-style, with a short
sliding window), so how much each head keeps depends on the input.

Paper: [razor.pdf](razor.pdf) (arXiv version pending).

## Gates

| Base model | Gate | Size |
|---|---|---|
| Qwen/Qwen3-1.7B | [Manalsu/razor-gate-Qwen3-1.7B](https://huggingface.co/Manalsu/razor-gate-Qwen3-1.7B) | 57 MB |
| Qwen/Qwen3-4B | [Manalsu/razor-gate-Qwen3-4B](https://huggingface.co/Manalsu/razor-gate-Qwen3-4B) | 91 MB |

## Results

All runs on 16 GB T4s. RULER and InfiniteBench scores are kvpress `string_match` averages.

| Benchmark | Model | Full cache | RAZOR (evicted) | Same gate, KVzap objective* | KVzip+ |
|---|---|---|---|---|---|
| RULER 4k (n=650, 5 seeds) | Qwen3-1.7B | 88.9 | 86.6 ± 0.6 (76%) | 79.6 | 86.5 (76%) |
| RULER 16k (n=195) | Qwen3-1.7B | 77.9 | 75.4 (81%) | 63.6 | 73.6 (76%) |
| RULER 4k (n=195) | Qwen3-4B | 94.3 | 93.9 (74%) | | |
| MATH-500 L4-5, compressed while reasoning (n=30) | Qwen3-1.7B | 66.7% | 73.3% (75%) | | |

KVzip+ needs a second pass over the context. RAZOR scores tokens in the normal forward pass.

\* Same architecture, data and hyperparameters, trained with KVzap's objective (regress KVzip+ scores). NVIDIA has not
released a KVzap gate for Qwen3-1.7B, so this is our own training, not their checkpoint.

Longest context on one 16 GB T4 (`bench_memory.py`, passkey retrieval, Qwen3-4B fp16, YaRN x4, chunked prefill):

| Tokens | Full cache peak GB | RAZOR peak GB / KV GB | RAZOR passkey |
|---|---|---|---|
| 31k | 15.0 (limit) | 9.6 / 0.72 | correct |
| 61k | OOM | 11.1 / 1.41 | correct |
| 96k | OOM | 12.6 / 2.22 | correct |
| 123k | OOM | 13.9 / 2.84 | correct |

At 29k tokens RAZOR decodes 2.9x faster than the full cache (7.5 vs 2.6 tokens/s). At 8k it is 7% slower, because of
the gate and the ragged-cache bookkeeping.

InfiniteBench at 100k tokens (Qwen3-4B, n=12 per task) gives the same scores as the full cache on passkey (0.67),
number_string (0.67) and kv_retrieval (0.08), and 0.67 vs 0.58 on longbook_choice.

Limits: tested only on Qwen3-1.7B and 4B. RAZOR is weak on common-word extraction (RULER cwe), which needs every
occurrence of a repeated token. MATH n=30 shows no loss, not a gain.

## Install

```bash
pip install razor-kv
```

## Use

```python
import razor_kv

pipe = razor_kv.pipeline("Qwen/Qwen3-4B")  # loads the model and the matching gate from Hugging Face
print(pipe(long_document, question="...")["answer"])  # evicted KV is freed, not masked
```

`threshold=` trades memory for accuracy (higher evicts more, default 0). `gate=` takes your own gate (HF repo id or
local directory). Batch size 1.

To use the press directly, with your own kvpress pipeline:

```python
from razor_kv import RaggedCache, enable, press

enable(model)  # per-head ragged attention
pipe(long_document, question="...", press=press("Qwen/Qwen3-4B"), cache=RaggedCache())
```

`DMSPress(LocalGatePress(path))` makes the same eviction decisions with kvpress's built-in masking, which saves no
memory. The RULER numbers were produced that way, through kvpress's `evaluation/evaluate.py` (`run_eval.py` registers
the gates). Beyond Qwen3's native 32k, load the model with YaRN factor 4; see `bench_memory.py`.

## Train a gate

```bash
git clone https://github.com/Manaslu-AI/razor && cd razor && pip install -e .
git clone --depth 1 https://github.com/NVIDIA/kvpress kvpress-src
python razor_kv/train_razor.py --model_name Qwen/Qwen3-1.7B --output_dir gates/razor --kvpress_src kvpress-src
```

## Files

- `razor_kv/train_razor.py`: trains a gate for a Hugging Face causal LM
- `razor_kv/press.py`: loads a gate into kvpress
- `razor_kv/ragged.py`: frees evicted KV (per-head ragged cache and attention, chunked prefill). `python -m razor_kv.ragged`
  checks that it matches kvpress masking to 1e-7
- `razor_kv/run_eval.py`: kvpress RULER / InfiniteBench evaluation with the gates registered
- `razor_kv/bench_memory.py`: longest context, peak memory and decode speed
- `razor_kv/eval_reason.py`: compress while reasoning, MATH-500
- `razor_kv/eval_rd.py`: one point on the rate-distortion curve (held-out KL vs memory kept)
- `razor_kv/gate_viz.py`: per-head keep rates and per-token decisions for one passage
- `razor_kv/build_kaggle.py`: each paper experiment as a Kaggle T4x2 job
- `razor_kv/test_razor_attention.py`: checks the eviction-aware training attention against a reference

## Citation

```bibtex
@misc{rawal2026razor,
  title  = {RAZOR: Learning What to Forget in the KV Cache with a Description-Length Price},
  author = {Rawal, Himansu},
  year   = {2026},
  note   = {Manaslu Labs. arXiv version pending},
  url    = {https://github.com/Manaslu-AI/razor}
}
```

## License

Apache-2.0. The gates are trained on top of Qwen3 (Apache-2.0) and use NVIDIA kvpress (Apache-2.0).
