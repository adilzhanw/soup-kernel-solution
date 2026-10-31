"""Kaggle T4 benchmark driver — self-contained, runs from /kaggle/working.

Clones the submission repo (harness/ + task/ + data/ + solution/), installs the exact
dependency set, then runs the official harness: base `soup train` and the same run with
solution.apply() (fused Triton cross-entropy/entropy/accuracy kernels). Prints
runs/result.json + runs/checks.json at the end.

Designed to finish inside Kaggle's 9h window (in practice ~30-60 min on a T4:
2 x 64 steps of Qwen2.5-1.5B 4-bit + downloads).

Usage: paste the whole file into one Kaggle notebook cell and run, or
    python run_kaggle.py
"""
import json
import os
import subprocess
import sys

REPO = "https://github.com/adilzhanw/soup-kernel-solution"
WORK = "/kaggle/working"
REPO_DIR = os.path.join(WORK, "repo")


def sh(cmd, **kw):
    print(f"\n$ {cmd}\n", flush=True)
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, **kw)
    if r.stdout:
        print(r.stdout[-4000:], flush=True)
    if r.stderr:
        print("STDERR:", r.stderr[-4000:], flush=True)
    return r


def main():
    # 1) clone the submission
    if not os.path.isdir(REPO_DIR):
        r = sh(f"git clone --depth 1 {REPO} {REPO_DIR}")
        if r.returncode != 0:
            sys.exit("clone failed")
    os.chdir(REPO_DIR)

    # 2) deps: torch/triton are preinstalled on the Kaggle T4 image; pin the rest
    sh(
        "pip install -q 'transformers<4.46' 'peft==0.12.0' 'bitsandbytes==0.43.3' "
        "'accelerate>=0.30' 'datasets>=2.20' 'huggingface_hub>=0.22' "
        "'sentencepiece' 'protobuf' 'soup-cli==0.77.0'"
    )
    sh(
        "python -c \"import torch, triton, transformers, peft, bitsandbytes; "
        "print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), "
        "torch.cuda.get_device_name(0)); print('triton', triton.__version__)\""
    )
    r = sh("python -c \"import torch; assert torch.cuda.is_available(), 'no GPU - set Accelerator to T4'\"")
    if r.returncode != 0:
        sys.exit("GPU is not visible: Session options -> Accelerator -> GPU T4")

    # 3) base run: plain soup train (harness measures wall time + peak memory)
    env_base = {k: v for k, v in os.environ.items() if k != "USE_OURS"}
    sh("python harness/run_task.py --config task/task.yaml --only base --timeout 7200",
       cwd=REPO_DIR, env=env_base)

    # 4) candidate run: same command, solution.apply() active (USE_OURS just documents intent;
    #    the harness switches sides via --only/--solution, which run_task.py handles)
    env_cand = dict(os.environ, USE_OURS="1")
    sh("python harness/run_task.py --config task/task.yaml --only solution --timeout 7200",
       cwd=REPO_DIR, env=env_cand)

    # 5) summary: metrics + ratios
    print("\n==================== RESULT =====================", flush=True)
    for tag in ("base_1", "solution_1"):
        p = os.path.join(REPO_DIR, "runs", tag, "metrics.json")
        if os.path.exists(p):
            m = json.load(open(p))
            keep = {k: m.get(k) for k in
                    ("exit_code", "steps", "median_step_s", "timed_loop_s",
                     "torch_max_allocated_mib", "own_kernel_ms", "triton_files")}
            print(f"\n--- {tag} ---\n{json.dumps(keep, indent=2)}")
        else:
            print(f"\n!! {p} missing")
    mb_p, ms_p = (os.path.join(REPO_DIR, "runs", t, "metrics.json") for t in ("base_1", "solution_1"))
    if os.path.exists(mb_p) and os.path.exists(ms_p):
        mb, ms = json.load(open(mb_p)), json.load(open(ms_p))
        if mb.get("timed_loop_s") and ms.get("timed_loop_s"):
            print(f"\nSPEEDUP (loop):      {mb['timed_loop_s'] / ms['timed_loop_s']:.3f}x")
        if mb.get("torch_max_allocated_mib") and ms.get("torch_max_allocated_mib"):
            print(f"PEAK ALLOC MiB:      base {mb['torch_max_allocated_mib']}  "
                  f"solution {ms['torch_max_allocated_mib']}")

    # 6) the 10-rule checks (needs the combined result.json from a full run; if we ran the
    #    --only sides separately, the jury's merged result is what check_rules.py grades)
    res = os.path.join(REPO_DIR, "runs", "result.json")
    if os.path.exists(res):
        print("\n--- result.json ---\n" + open(res).read())
    chk = sh("python harness/check_rules.py")
    if os.path.exists(os.path.join(REPO_DIR, "runs", "checks.json")):
        print("\n--- checks.json ---\n" + open(os.path.join(REPO_DIR, "runs", "checks.json")).read())
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
