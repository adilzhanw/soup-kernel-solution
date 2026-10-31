# Soup Kernel Challenge — Fused CE / Entropy / Accuracy Triton Kernels

**Decentrathon 6.0** · Soup `soup-cli 0.77.0` · `Qwen/Qwen2.5-1.5B-Instruct` (4-bit NF4) · LoRA r=16 · **Triton**

## What we did

Base Soup (TRL SFT + `RewindSFTTrainer`) computes, on **every training step**, after the
model returns the full `[B, L, 151936]` logits:

| Operation | Where | Cost |
|-----------|-------|------|
| cross-entropy `log_softmax` (fwd) | `ForCausalLMLoss` | O(V) in **fp32** + materialized `[B,L,V]` fp32 |
| cross-entropy backward (re-softmax) | autograd | **another** full O(V) softmax, fp32 |
| per-row Shannon entropy (fwd) | Soup metric | O(V) reduce, fp32 |
| argmax token accuracy (fwd) | Soup metric | O(V) reduce |

For batch 4 × seq 512 that is `4·512·151936 ≈ 311 M` floats touched **four times** per step,
in fp32, with two full materialized `[B,L,V]` fp32 intermediates (≈ 1.2 GB each). On the T4
this is the loss's share of wall time plus a chunk of the elementwise/dtype-cast bucket, and
it dominates the logits' activation memory.

### The fix

Two hand-written `@triton.jit` kernels replace all four reductions:

1. **`fused_softmax_fwd_kernel`** — one program per row; a two-loop streaming (online)
   reduction keeps the running max, so the kernel is numerically stable for any logit scale
   and never needs a separate global-max pass. One pass produces: the cross-entropy **sum**
   at the target (the trainer divides by `num_items_in_batch`, exactly as the base), the row
   **entropy** in nats, and the **argmax accuracy** numerator/denominator.
2. **`fused_softmax_bwd_kernel`** — reads only `(logits, labels)` and writes
   `grad[:, v] = p[:, v] − 1[v == target]` directly (rows with label `-100` get zero
   gradient, matching `ignore_index`). The backward is mathematically identical to
   `F.cross_entropy`, with **no** fp32 `[B,L,V]` softmax materialization.

The loss value is numerically equivalent to `torch.nn.functional.cross_entropy`
(measured on RTX 5070, full task shapes `[4, 512, 151936]`, fp16 **and** fp32:
loss rel. diff `< 8e-7`, max gradient diff `< 2.4e-7`). Only the fp32 summation order
differs, which is far inside the task's parity gates.

### Why this is the right target

- It is the **single hottest non-attention region** of a 1.5B model at seq 512: the
  151k-vocab softmax/reduce budget.
- It removes **two full fp32 `[B,L,V]` tensors** (≈ 1.2 GB each) from the live graph —
  the source of the peak-memory reduction.
- It does not touch attention, the model, LoRA, the loss *value*, or any hyperparameter —
  only how the per-step reductions are executed. `harness/` and `task/` are byte-identical
  to the starter.

## Files

```
solution/fused_loss.py    # the two @triton.jit kernels + the torch.autograd.Function
solution/__init__.py      # apply(): monkey-patches Qwen2 loss_function + Soup metrics hooks
solution/test_kernels.py  # 7 correctness tests vs plain PyTorch (loss, grad, H, acc, -100 mask)
harness/                  # Soup harness — unmodified
task/                     # task.yaml + prepare_data.py — unmodified
data/                     # fixed train/eval split (seed 13)
kaggle_notebook.ipynb     # the T4 benchmark, cell by cell
run_kaggle.py             # same benchmark as one self-contained script
```

## Integration

`solution.apply()` (called by `harness/_train.py --solution`, before the trainer starts):

1. replaces `Qwen2ForCausalLM.loss_function` with the fused kernel (it mirrors the base
   shift: `logits[..., :-1]` vs `labels[..., 1:]`, `reduction="sum"`);
2. wraps `SFTTrainer.compute_loss` so Soup's `entropy` / `mean_token_accuracy` metrics are
   read from the fused kernel's outputs instead of re-running two full softmax passes.

No change to the model, optimizer, scheduler, LoRA config, batch size, or sequence length.

## How to run (Kaggle T4)

Open `kaggle_notebook.ipynb` (or paste `run_kaggle.py` into one cell), set
**Session options → Accelerator → GPU T4**, Run All:

1. clone this repo, install the pinned deps (`soup-cli==0.77.0`, `transformers<4.46`,
   `peft==0.12.0`, `bitsandbytes==0.43.3`, …);
2. `python harness/run_task.py --config task/task.yaml --only base` — plain `soup train`,
   64 steps, timed from outside;
3. the same with `--only solution` — `solution.apply()` active;
4. print `runs/*/metrics.json` + `runs/result.json`, then `python harness/check_rules.py`
   for the 10-rule pass/fail table.

Local sanity (any CUDA GPU):

```
python -m pytest solution -q        # 7 tests, < 2 s
python harness/run_task.py --config task/task.yaml   # base + solution back-to-back
python harness/check_rules.py
```

## Result (T4)

| Metric | Base | Candidate | Δ |
|--------|------|-----------|---|
| Wall time (64 steps) | _TBD — T4 run pending_ | _TBD_ | **≥ 1.10× required** |
| Peak GPU memory | _TBD_ | _TBD_ | **−25 % required** |
| Fused-kernel share of step time | _TBD_ | _TBD_ | **≥ 5 % required** |
| Eval CE / acc parity | _TBD_ | _TBD_ | within gates |

The authoritative numbers come from the T4 run (the local 8 GB 5070-LP cannot host the
task's batch 4 / seq 512 — it OOMs on the base path itself; see *Local environment* below).
Unit-level parity and the kernel's standalone cost **are** measured locally and listed in
*Local verification*.

## Checks (from the harness)

| # | Check | Status |
|---|-------|--------|
| 1 | Runs to completion (64 steps, no NaN) | pending T4 (local GPU too small — see below) |
| 2 | `runs/{base,solution}*/run.log` exist | pending T4 |
| 3 | `runs/{base,solution}*/metrics.json` exist | pending T4 |
| 4 | `check_rules.py` → kernels `OWN`, no banned libs | ✅ static scan: 2 `@triton.jit` in `solution/`, zero banned imports, similarity to the shipped example ≪ 0.80 |
| 5 | Custom kernels present, ≥ 5 % of time | ✅ present; the harness `TritonWatch` counts & times them by name in every run (verified live: `fused_softmax_fwd/bwd_kernel` attributed to `solution/fused_loss.py`); share pending T4 |
| 6 | Base vs candidate metrics within `torch.allclose` gates | ✅ unit-level: loss rel `< 8e-7`, grad max diff `< 2.4e-7` (fp16 & fp32, full `[4,512,151936]` shapes, `-100` masking) |
| 7 | Wall-time speedup ≥ 1.10 | pending T4 |
| 8 | Eval CE / acc parity | pending T4 |
| 9 | Peak memory reduction ≥ 25 % | pending T4 |
| 10 | 2–4 min video: profile + run + result | ⬜ (plan in `video_plan.md`) |

## Local verification (RTX 5070 Laptop, 8 GB, torch 2.9 / triton 3.5)

- `pytest solution`: **7/7 passed** — loss / gradient / entropy / accuracy vs plain
  PyTorch, fp16 + fp32, shapes up to `[4, 512, 151936]`, `-100` padding masks.
- Harness dry-run with `--solution`: `apply()` executes, both kernels are detected and
  attributed as **OWN** (`triton_files: {fused_softmax_*: solution/fused_loss.py}`),
  zero banned modules loaded.

## Local environment note (why the end-to-end number is T4-only)

The dev machine's GPU (8 GB) cannot run the fixed task at batch 4 / seq 512: the **base**
path itself peaks at ~7.2 GB allocated — model + optimizer + the two fp32 `[B,L,V]` loss
temporaries exceed the card, and the run dies on step 2's `lm_head` projection with
`CUDA OOM` (crash bundle in `.soup-crashes/`). This is expected — the task is defined for a
16 GB T4, where the same base run fits. The kernel's correctness and standalone cost are
therefore verified at unit level locally; wall-time / peak-memory / kernel-share come from
the T4 notebook above.

## Reproducibility

- Fixed data seed `13`, fixed model revision (Hub commit bound in the run log), fixed LoRA
  (r=16, α=32, dropout 0.05); `seed=1`, `data_seed` unset — exactly as `task/task.yaml`.
- `PYTORCH_ALLOC_CONF=expandable_segments:True` on **both** sides (harness setting).
- Dependencies: `torch`, `triton`, `transformers<4.46`, `peft==0.12.0`,
  `bitsandbytes==0.43.3`, `soup-cli==0.77.0`.
- Banned: `unsloth`, `liger_kernel`, `xformers`, `flash_attn`, `torchao`, `apex`,
  `transformer_engine`, `triton.ops`, prebuilt kernel repos — **none used** (the harness
  additionally blocks `unsloth`/`unsloth_zoo` imports at run start).
