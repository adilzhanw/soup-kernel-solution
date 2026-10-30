# Soup Kernel Challenge — Fused CE / Entropy / Accuracy Triton Kernels

**Decentrathon 6.0** · Soup `0.77.0` · `Qwen/Qwen2.5-1.5B-Instruct` (4-bit NF4) · LoRA r=16 · **Triton**

## What we did

Base Soup computes **four full-softmax reductions** over the `[B, L, 151936]`
vocabulary axis on every training step:

| Operation | Where | Cost |
|-----------|-------|------|
| `log_softmax` (fwd) | loss | O(V) softmax + materialized `[B,L,V]` fp32 |
| `log_softmax` (bwd) | loss | **another** O(V) softmax |
| `entropy` (fwd) | `H = -Σ p·log p` | O(V) reduce |
| `acc` (fwd) | `softmax.max` | O(V) reduce |

For batch 4 × seq 512 that is `4·512·151936 ≈ 311 M` floats touched **four
times** per step, in fp32, with two full materialized `[B,L,V]` intermediates.
On a T4 (16 GB, ~8 TB/s HBM) this is a dominant share of wall time **and** a
major contributor to peak memory.

### The fix

Two hand-written Triton kernels replace all four reductions:

1. **`fused_ce_h_acc_fwd`** — reads the raw logits once and produces
   `(loss, entropy, accuracy, logsumexp, max)` in a **single** fused pass.
   It is *online*: a two-loop streaming reduction keeps the running max so it
   never needs a global max first and is numerically stable for any logit scale.
   The log-sum-exp is kept so the backward is exact with **zero** recomputed
   softmax over the vocab.
2. **`fused_ce_h_acc_bwd`** — reads only `(logits, logsumexp, max)` and writes
   `dlogits = (softmax − onehot) · dloss − dH·(softmax·(logit − logsumexp) − p)`
   directly. No `[B,L,V]` fp32 softmax materialization.

The loss value is mathematically **identical** to `torch.nn.functional
cross_entropy` (verified to `< 1e-4` abs, `< 1e-5` grad); only the
reduction order differs (fp32 atomic/online sum), which is within the
task's `atol=1e-3` gate and the harness's `torch.allclose`.

### Why this is the right target

- It is the **single hottest non-attention region** on a 1.5B model at seq 512:
  the 151k-vocab softmax/reduce dominates the per-step elementwise + reduce
  budget.
- It removes **two full materialized `[B,L,V]` fp32 tensors** (≈ 1.25 GB each
  at our shapes) from the live graph, which is where the −25 % peak-memory
  target comes from.
- It does not touch attention, the model, LoRA, the loss *value*, or any
  hyperparameter — only the way the per-step reductions are executed.

## Files

```
solution/fused_loss.py   # the two @triton.jit kernels + torch wrappers
solution/__init__.py     # apply(): monkey-patches Soup's per-step CE/H/acc
solution/test_kernels.py # correctness + memory micro-benchmark
harness/                 # Soup harness (run_task.py, check_rules.py, …) — unmodified
task/                    # task.yaml + data prep — unmodified
data/                    # train/eval split (fixed seed 13)
```

## Integration

`solution.apply()` hooks the trainer right after `model/loss_fn` are created
and wraps `loss_fn` so that, when `labels` are present, it:

1. runs the model,
2. computes `(loss, H, acc)` with the fused kernels,
3. returns `{"loss": loss, "acc": acc}` (plus `H` for the logging hook)
   — the exact contract Soup's trainer expects.

No change to the model, optimizer, scheduler, LoRA config, batch, or sequence.

## How to run

The task ships a ready Kaggle notebook. Fill `REPO_URL` (or run from a local
clone) on a **T4 / XLA**:

```
python harness/run_task.py --config task/task.yaml --mode bench --tag base
python harness/run_task.py --config task/task.yaml --mode bench --tag candidate
```

`check_rules.py` reports `OWN` kernels and confirms no banned library is
imported. `test_kernels.py` asserts numerical parity against PyTorch and
measures the memory/time of the fused path vs. the naive one.

## Result (T4, XLA)

| Metric | Base | Candidate | Δ |
|--------|------|-----------|---|
| Training wall time | _TBD_ | _TBD_ | **≥ 1.10× required** |
| Peak GPU memory | _TBD_ | _TBD_ | **−25 % required** |
| Triton share of time | _TBD_ | _TBD_ | **≥ 5 % required** |
| Eval CE / acc | _TBD_ | _TBD_ | parity |

_Filled after the free-T4 benchmark run. `runs_base/` and `runs_local/`
contain the local (5070) regression + micro-benchmark artifacts; the
authoritative numbers come from the T4 run._

## Checks (from `task.yaml`)

| # | Check | Status |
|---|-------|--------|
| 1 | Runs to completion (64 steps, no NaN) | ✅ (local) / pending T4 |
| 2 | `runs/{base,candidate}/run.log` exist | pending T4 |
| 3 | `runs/{base,candidate}/metrics.json` exist | pending T4 |
| 4 | `harness/check_rules.py` → `OWN`, no banned libs | ✅ |
| 5 | Custom `@triton.jit` kernels present, ≥ 5 % of time | ✅ present / share pending T4 |
| 6 | `torch.allclose` base vs candidate metrics | ✅ (local parity) |
| 7 | Wall-time speedup ≥ 1.10 | pending T4 |
| 8 | Eval CE / acc parity | pending T4 |
| 9 | Peak memory reduction ≥ 25 % | pending T4 |
| 10 | 2–4 min video: profile + run + result | ⬜ |

## Reproducibility

- Fixed data seed `13`, fixed model, fixed LoRA (r=16, α=32, dropout 0.05).
- `seed=42` in `task.yaml`, `no_cuda_mps`.
- All dependencies: `transformers`, `peft`, `bitsandbytes`, `triton`, `torch`.
- Banned: `unsloth`, `liger_kernel`, `xformers`, `flash_attn`, `torchao`,
  `apex`, `transformer_engine`, `bitsandbytes` (for kernel tricks), `triton.ops`,
  prebuilt kernel repos — **none used**.
