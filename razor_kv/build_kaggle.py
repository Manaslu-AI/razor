"""Bundle razor_kv/*.py into a single Kaggle script that runs two GPU tracks in parallel.

python razor_kv/build_kaggle.py sanity|full   -> writes kaggle_razor/main.py + kernel-metadata.json
then: kaggle kernels push -p kaggle_razor --accelerator NvidiaTeslaT4
"""
import base64
import glob
import json
import os
import sys

MODE = sys.argv[1] if len(sys.argv) > 1 else "sanity"
M = "Qwen/Qwen3-1.7B"
OWNER = os.environ.get("KAGGLE_OWNER", "")  # your Kaggle username; modes that reuse gates expect a dataset OWNER/razor-gates
EVAL = "cd /kaggle/kvpress-src/evaluation && python /kaggle/working/occam/razor_kv/run_eval.py --model {m} --dataset ruler --data_dir 4096 --fraction {f} --output_dir /kaggle/working/results --press_name {p} --compression_ratio {cr}"
TRAIN_R = "python /kaggle/working/occam/razor_kv/train_razor.py --model_name {m} --output_dir /kaggle/working/gates/razor --kvpress_src /kaggle/kvpress-src --n_train_per_subset {n}"
TRAIN_K = "cd /kaggle/kvpress-src && python kvzap/train.py --model_name {m} --output_dir /kaggle/working/gates/kvzap --n_train_per_subset {n} --n_tokens {nt} --max_epochs 15"

after = []  # stages run after both tracks, with every GPU visible
if MODE == "sanity":
    f, n, nt, crs = 0.003, 3, 50, [0.75]
    track0 = [TRAIN_R.format(m=M, n=n)] + [EVAL.format(m=M, f=f, p="razor_head", cr=c) for c in crs]
    track1 = [TRAIN_K.format(m=M, n=n, nt=nt)] + [EVAL.format(m=M, f=f, p="kvzap_ours_head", cr=c) for c in crs]
elif MODE == "ablate":  # does each part of the thesis matter? (same data, same eval as the full run)
    f, n = 0.03, 200
    tr = TRAIN_R.format(m=M, n=n).replace("/gates/razor", "/gates/{g}")
    track0 = [tr.format(g="razor_nomdl") + " --use_mdl False", EVAL.format(m=M, f=f, p="razor_nomdl_head", cr=0.75),
              tr.format(g="razor_soft") + " --hard False", EVAL.format(m=M, f=f, p="razor_soft_head", cr=0.75)]
    track1 = [tr.format(g="razor_long") + " --max_len 2048 --min_tokens 1500 --max_tokens 2600",EVAL.format(m=M, f=f, p="razor_long_head", cr=0.75),
              EVAL.format(m=M, f=f, p="razor_long_head", cr=0.5)]
elif MODE == "recon":  # query-agnostic objective; same data/eval as the full run
    f, n = 0.03, 200
    tr = TRAIN_R.format(m=M, n=n).replace("/gates/razor", "/gates/{g}") + " --objective recon"
    track0 = [tr.format(g="razor_recon")] + [EVAL.format(m=M, f=f, p="razor_recon_head", cr=c) for c in (0.75, 0.5)]
    track1 = [tr.format(g="razor_recon15") + " --target_keep 0.15"] + [EVAL.format(m=M, f=f, p="razor_recon15_head", cr=c) for c in (0.75, 0.5)]
elif MODE == "topk":  # KL only on the 5% hardest positions; same data/eval as the full run
    f, n = 0.03, 200
    tr = TRAIN_R.format(m=M, n=n).replace("/gates/razor", "/gates/{g}") + " --kl_top 0.05"
    track0 = [tr.format(g="razor_top")] + [EVAL.format(m=M, f=f, p="razor_top_head", cr=c) for c in (0.75, 0.5)]
    track1 = [tr.format(g="razor_recontop") + " --objective recon"] + [EVAL.format(m=M, f=f, p="razor_recontop_head", cr=c) for c in (0.75, 0.5)]
elif MODE == "seed":  # is the original RAZOR result stable, and does more training help?
    f, n = 0.03, 200
    tr = TRAIN_R.format(m=M, n=n).replace("/gates/razor", "/gates/{g}")
    track0 = [tr.format(g="razor_s1") + " --seed 1"] + [EVAL.format(m=M, f=f, p="razor_s1_head", cr=c) for c in (0.75, 0.5)]
    track1 = [tr.format(g="razor_ep2") + " --epochs 2"] + [EVAL.format(m=M, f=f, p="razor_ep2_head", cr=c) for c in (0.75, 0.5)]
elif MODE == "dms":  # the thesis's own inference mode: keep a token iff its gate says so (threshold), vs official KVzap
    f = 0.03
    DMS = EVAL.replace("--compression_ratio {cr}", "--threshold {t}")
    track0 = [DMS.format(m=M, f=f, p="razor_dms", t=t) for t in (0.0, -1.0, 1.0)] + [DMS.format(m=M, f=f, p="razor_s1_dms", t=0.0)]
    track1 = [EVAL.format(m=M, f=f, p="kvzap_mlp_head", cr=c) for c in (0.75, 0.5)] + [DMS.format(m=M, f=f, p="kvzap_mlp", t=t) for t in (-4.0, -3.0)]
elif MODE == "fair":  # is RAZOR's threshold-mode win from its training, or just from per-head adaptive budgets?
    f = 0.03
    DMS = EVAL.replace("--compression_ratio {cr}", "--threshold {t}")
    track0 = [DMS.format(m=M, f=f, p="kvzap_ours_dms", t=t) for t in (-4.0, -5.0, -6.0, -3.0)]
    track1 = [EVAL.format(m=M, f=f, p=p, cr=0.75) for p in ("razor_ada", "kvzap_ours_ada", "fastkvzip", "kvzip_plus")]
elif MODE == "confirm":  # 3x larger sample, matched ~76% eviction: RAZOR vs same-arch KVzap vs best expensive baseline
    f = 0.1
    DMS = EVAL.replace("--compression_ratio {cr}", "--threshold {t}")
    track0 = [DMS.format(m=M, f=f, p="razor_dms", t=0.0), DMS.format(m=M, f=f, p="kvzap_ours_dms", t=-4.5)]
    track1 = [EVAL.format(m=M, f=f, p="kvzip_plus", cr=0.76), EVAL.format(m=M, f=f, p="no_press", cr=0.0)]
elif MODE == "long":  # RULER 16k (4x the context the gate was trained on), matched ~76% eviction
    f = 0.03
    L = EVAL.replace("--data_dir 4096", "--data_dir 16384") + """ --model_kwargs '{{"dtype": "float16"}}'"""  # T4: bf16 -> math kernel -> OOM
    DMS = L.replace("--compression_ratio {cr}", "--threshold {t}")
    track0 = [DMS.format(m=M, f=f, p="razor_dms", t=0.0), L.format(m=M, f=f, p="no_press", cr=0.0)]
    track1 = [L.format(m=M, f=f, p="kvzip_plus", cr=0.76), DMS.format(m=M, f=f, p="kvzap_ours_dms", t=-4.5)]
elif MODE == "100k":  # InfiniteBench truncated to 100k tokens. RAZOR/KVzap free evicted KV (ragged.py) -> one T4 each, in
    # parallel; full-cache baselines need ~11.5 GB of KV alone -> run afterwards with the model split over both T4s
    I = (EVAL.replace("&& python", "&& RAZOR_YARN=4 python").replace("--dataset ruler --data_dir 4096", "--dataset infinitebench --data_dir {task} --max_context_length 100000")
         + """ --model_kwargs '{{"dtype": "float16"}}'""")
    DMS = I.replace("--compression_ratio {cr}", "--threshold {t}")
    tasks = {"passkey": 0.014, "number_string": 0.014, "kv_retrieval": 0.016, "longbook_choice_eng": 0.035}  # ~8 each (T4: ~200 s/prefill)
    track0 = [DMS.format(m=M, f=f, task=task, p="razor_ragged", t=0.0) for task, f in tasks.items()]
    track1 = [DMS.format(m=M, f=f, task=task, p="kvzap_ours_ragged", t=-4.5) for task, f in tasks.items()]
    after = [c for task, f in tasks.items() for c in [I.format(m=M, f=f, task=task, p="no_press", cr=0.0),
                                                      I.format(m=M, f=f, task=task, p="kvzip_plus", cr=0.76)]]
elif MODE == "100k4b":  # the 1.7B model fails 100k even uncompressed (passkey 0/8) -> Qwen3-4B, which passed 123k.
    # RAZOR on ONE T4. Full-cache reference = same code with nothing evicted (threshold -1e9; equal to plain prefill
    # to 1e-7, see ragged.py), chunked, model split over both T4s: plain no_press prefill would not fit 100k at 4B.
    M4 = "Qwen/Qwen3-4B"
    I = (EVAL.replace("&& python", "&& RAZOR_YARN=4 RAZOR_CHUNK=4096 python").replace("--dataset ruler --data_dir 4096", "--dataset infinitebench --data_dir {task} --max_context_length 100000")
         + """ --model_kwargs '{{"dtype": "float16"}}'""").replace("--compression_ratio {cr}", "--threshold {t}")
    tasks = {"passkey": 0.02, "number_string": 0.02, "longbook_choice_eng": 0.052, "kv_retrieval": 0.024}  # ~12 each
    track0 = [I.replace("&& RAZOR_YARN", "&& CUDA_VISIBLE_DEVICES=0 RAZOR_YARN").format(m=M4, f=f, task=task, p="razor4b_ragged", t=0.0) for task, f in tasks.items()]
    track1 = []
    after = [I.format(m=M4, f=f, task=task, p="razor4b_ragged", t=-1e9) for task, f in tasks.items()]
elif MODE == "mem":  # longest context on ONE 16 GB T4: full cache vs RAZOR with real deletion (ragged.py)
    B = "cd /kaggle/working && CUDA_VISIBLE_DEVICES=0 python /kaggle/working/occam/razor_kv/bench_memory.py --gate /kaggle/working/gates/razor --methods {ms} --lengths {ls}"
    track0 = [B.format(ms="razor,full", ls="32000,64000"), B.format(ms="razor", ls="100000,128000")+ " --out /kaggle/working/bench_memory_long.json",
              B.format(ms="full", ls="100000,128000") + " --out /kaggle/working/bench_memory_full.json"]
    track1 = []
elif MODE == "4b":  # Qwen3-4B at 128k on ONE T4: its full KV (~19 GB) is bigger than the GPU. Train its gate (GPU0)
    # while GPU1 runs the no_press RULER-4k baseline; then memory/passkey bench and RAZOR on the same RULER-4k sample
    M4, f = "Qwen/Qwen3-4B", 0.03
    E = EVAL + """ --model_kwargs '{{"dtype": "float16"}}'"""
    B = ("cd /kaggle/working && CUDA_VISIBLE_DEVICES=0 python /kaggle/working/occam/razor_kv/bench_memory.py --model_name " + M4 +
         " --gate /kaggle/working/gates/razor4b --chunk 4096 --methods {ms} --lengths {ls} --out /kaggle/working/bench_4b_{o}.json")
    track0 = [TRAIN_R.format(m=M4, n=200).replace("/gates/razor", "/gates/razor4b"),
              B.format(ms="razor", ls="32000,64000,100000,128000", o="razor"), B.format(ms="full", ls="16000,32000,64000", o="full")]
    track1 = [E.format(m=M4, f=f, p="no_press", cr=0.0)]
    after = [E.replace("--compression_ratio {cr}", "--threshold {t}").format(m=M4, f=f, p="razor4b_dms", t=0.0)]
elif MODE == "seeds":  # paper: mean +- std over 5 seeds at the main-table setting (RULER-4k n=650, threshold 0).
    # seed 0 = confirm run (86.7); seed 1 gate comes from the dataset; seeds 2-4 trained here (~40 min each)
    f = 0.1
    DMS = EVAL.replace("--compression_ratio {cr}", "--threshold {t}")
    tr = TRAIN_R.format(m=M, n=200).replace("/gates/razor", "/gates/{g}")
    track0 = [tr.format(g="razor_s2") + " --seed 2", DMS.format(m=M, f=f, p="razor_s2_dms", t=0.0),
              DMS.format(m=M, f=f, p="razor_s1_dms", t=0.0)]
    track1 = [tr.format(g="razor_s3") + " --seed 3", DMS.format(m=M, f=f, p="razor_s3_dms", t=0.0),
              tr.format(g="razor_s4") + " --seed 4", DMS.format(m=M, f=f, p="razor_s4_dms", t=0.0)]
elif MODE == "speed":  # decode tokens/s with preallocated ragged buffers, Qwen3-4B on one T4, full vs RAZOR
    B = ("cd /kaggle/working && CUDA_VISIBLE_DEVICES=0 python /kaggle/working/occam/razor_kv/bench_memory.py --model_name Qwen/Qwen3-4B"
         " --gate /kaggle/working/gates/razor4b --chunk 4096 --methods {ms} --lengths {ls} --out /kaggle/working/speed_{o}.json")
    track0 = [B.format(ms="full,razor", ls="8000,16000,30000", o="both"), B.format(ms="razor", ls="64000,120000", o="long")]
    # GPU1: soft vs hard gates under THRESHOLD inference (IndexMem++ argues STE is biased). Hard reference: razor_dms
    # t=0 -> 86.4 @ 75.9% evicted on this exact sample (f=0.03). Sweep t so the soft gate is compared at matched eviction.
    DMS = EVAL.replace("--compression_ratio {cr}", "--threshold {t}")
    track1 = [TRAIN_R.format(m=M, n=200).replace("/gates/razor", "/gates/razor_soft") + " --hard False"] + [
        DMS.format(m=M, f=0.03, p="razor_soft_dms", t=t) for t in (0.0, 1.0, -1.0)]
elif MODE == "rd":  # rate-distortion curve of the KV cache: held-out KL vs memory kept, gates trained at keep targets
    # 0.1/0.2/0.5 (+ the existing 0.3 gate swept over tau). Paper prediction: final dual price lambda = -dKL/dkeep.
    E = ("mkdir -p /kaggle/working/rd && cd /kaggle/working && python /kaggle/working/occam/razor_kv/eval_rd.py"
         " --kvpress_src /kaggle/kvpress-src --gate_dir /kaggle/working/gates/{g} --taus {t} --out /kaggle/working/rd/rd_{g}.json")
    tr = TRAIN_R.format(m=M, n=200).replace("/gates/razor", "/gates/{g}") + " --target_keep {r}"
    track0 = [E.format(g="razor", t="0,-1,1,-2,2"), tr.format(g="rho0.1", r=0.1), E.format(g="rho0.1", t="0"),
              tr.format(g="rho0.5", r=0.5), E.format(g="rho0.5", t="0")]
    track1 = [tr.format(g="rho0.2", r=0.2), E.format(g="rho0.2", t="0")]
elif MODE == "reason2":  # gate trained on reasoning traces; 30 problems, 8k cap (6k truncated ~45% of no_press answers)
    R = "python /kaggle/working/occam/razor_kv/eval_reason.py --press {p} --threshold {t} --n 30 --max_new_tokens 8000 --model_name " + M
    tr = TRAIN_R.format(m=M, n=260).replace("/gates/razor", "/gates/razor_math") + " --data openr1 --min_tokens 1500 --max_len 2048"
    track0 = [tr, R.format(p="razor_math", t=0.0)]
    track1 = [R.format(p="no_press", t=0.0), R.format(p="razor", t=0.0)]
elif MODE == "reason":  # compress the KV cache WHILE a thinking model reasons (MATH-500 level 4-5, 60 problems)
    R = "python /kaggle/working/occam/razor_kv/eval_reason.py --press {p} --threshold {t} --n 60 --model_name " + M
    track0 = [R.format(p="no_press", t=0.0), R.format(p="razor", t=0.0)]
    track1 = [R.format(p="kvzap_ours", t=-4.5), R.format(p="razor", t=1.0)]
else:
    f, n, nt, crs = 0.03, 200, 40, [0.75, 0.5]
    track0 = [TRAIN_R.format(m=M, n=n)] + [EVAL.format(m=M, f=f, p="razor_head", cr=c) for c in crs] + [
        EVAL.format(m=M, f=f, p="no_press", cr=0.0)] + [
        EVAL.format(m=M, f=f, p=p, cr=c) for p in ["streaming_llm", "snapkv"] for c in crs]
    track1 = [TRAIN_K.format(m=M, n=n, nt=nt)] + [EVAL.format(m=M, f=f, p="kvzap_ours_head", cr=c) for c in crs] + [
        EVAL.format(m=M, f=f, p=p, cr=c) for p in ["expected_attention", "kvzip"] for c in crs]

files = {p: base64.b64encode(open(p, "rb").read()).decode() for p in glob.glob("razor_kv/*.py")}

main = f'''import base64, os, subprocess, sys, threading, time
FILES = {files!r}
for p, b in FILES.items():
    q = "/kaggle/working/occam/" + p
    os.makedirs(os.path.dirname(q), exist_ok=True)
    open(q, "wb").write(base64.b64decode(b))

def sh(cmd, env=None, log=None):
    t0 = time.time()
    with open(log or os.devnull, "w") as fh:
        r = subprocess.run(cmd, shell=True, env=env, stdout=fh, stderr=subprocess.STDOUT)
    tail = open(log).read()[-1500:] if log else ""
    print(f"[{{'OK' if r.returncode == 0 else 'FAIL'}} {{time.time()-t0:.0f}}s] {{cmd[:160]}}\\n{{tail if r.returncode else ''}}", flush=True)
    return r.returncode

sh("pip install -q 'transformers>=4.56,<5.3' 'fire>=0.7' skorch scikit-learn rouge nltk scipy fuzzywuzzy jieba bert-score", log="/kaggle/working/pip.log")
sh("git clone -q --depth 1 https://github.com/NVIDIA/kvpress /kaggle/kvpress-src && pip install -q --no-deps -e /kaggle/kvpress-src", log="/kaggle/working/clone.log")
sh("nvidia-smi --query-gpu=name,memory.total --format=csv; python -c 'import transformers,torch;print(transformers.__version__,torch.__version__)'", log="/kaggle/working/env.log")
print(open("/kaggle/working/env.log").read(), flush=True)

import glob as _g, shutil
for c in _g.glob("/kaggle/input/**/config.json", recursive=True):  # pre-trained gates from the razor-gates dataset
    g = os.path.basename(os.path.dirname(c))
    shutil.copytree(os.path.dirname(c), "/kaggle/working/gates/" + ("kvzap/mlp" if g == "kvzap_ours" else g), dirs_exist_ok=True)
    print("gate from dataset:", c, flush=True)
TRACKS = {[track0, track1]!r}
ngpu = int(subprocess.run("nvidia-smi -L | wc -l", shell=True, capture_output=True, text=True).stdout.strip() or 1)
os.makedirs("/kaggle/working/logs", exist_ok=True)

def run_track(i, cmds, gpu):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
               PYTHONPATH="/kaggle/working/occam:/kaggle/kvpress-src", RAZOR_GATE="/kaggle/working/gates/razor", KVZAP_GATE="/kaggle/working/gates/kvzap/mlp",
               **{{"GATE_" + g.upper(): "/kaggle/working/gates/" + g for g in ["razor_nomdl", "razor_soft", "razor_long", "razor_recon", "razor_recon15", "razor_top", "razor_recontop", "razor_s1", "razor_s2", "razor_s3", "razor_s4", "razor_ep2", "razor_math", "razor4b"]}})
    for j, c in enumerate(cmds):
        sh(c, env=env, log=f"/kaggle/working/logs/t{{i}}_{{j}}.log")

if ngpu >= 2 and all(TRACKS):
    ts = [threading.Thread(target=run_track, args=(i, c, i)) for i, c in enumerate(TRACKS)]
    [t.start() for t in ts]; [t.join() for t in ts]
else:
    for i, c in enumerate(TRACKS):  # one track -> it gets every GPU (device_map=auto splits the model)
        run_track(i, c, ",".join(map(str, range(ngpu))))

run_track(2, {after!r}, ",".join(map(str, range(ngpu))))

import glob, json
for f in sorted(glob.glob("/kaggle/working/results/*/metrics.json")):
    m = json.load(open(f)); avg = m if isinstance(m, (int, float)) else sum(v["string_match"] for v in m.values()) / len(m)
    print(f"RESULT {{os.path.basename(os.path.dirname(f))}} avg={{avg:.2f}}", flush=True)
for f in sorted(glob.glob("/kaggle/working/reason/*.json")):
    r = json.load(open(f)); x = r["rows"]
    print(f"REASON {{r['press']}} n={{len(x)}} acc={{sum(y['correct'] for y in x)/len(x):.3f}} evicted={{sum(y['evicted'] for y in x)/len(x):.3f}} partial={{r.get('partial', False)}}", flush=True)
print("ALL DONE", flush=True)
'''

os.makedirs("kaggle_razor", exist_ok=True)
open("kaggle_razor/main.py", "w").write(main)
kid = "razor-kvpress" if MODE in ("sanity", "full") else f"razor-kvpress-{MODE}"  # separate kernel: pushing replaces outputs
json.dump({"id": f"{OWNER}/{kid}", "title": kid, "code_file": "main.py", "language": "python",
           "kernel_type": "script", "is_private": "true", "enable_gpu": "true", "enable_tpu": "false",
           "enable_internet": "true", "dataset_sources": [f"{OWNER}/razor-gates"] if MODE in ("dms", "fair", "confirm", "reason", "reason2", "long", "100k", "mem", "4b", "100k4b", "speed", "seeds", "rd") else [], "competition_sources": [], "kernel_sources": [],
           "model_sources": []}, open("kaggle_razor/kernel-metadata.json", "w"), indent=2)
print(f"built kaggle_razor/main.py ({MODE}): track0={len(track0)} track1={len(track1)} stages")
