# Fused CE / Entropy / Accuracy Triton Kernels
### Soup Kernel Challenge · Decentrathon 6.0

---

## 1. The task
- Model: `Qwen2.5-1.5B-Instruct`, 4-bit NF4, LoRA r=16
- Setup: batch 4 · seq 512 · 1 epoch (64 steps) · fp16 · T4 16 GB
- Goal: **cut training time and GPU memory** with our own `@triton.jit` kernels
- Constraints: no unsloth / liger / xformers / flash_attn / torchao; no changes to
  model, LoRA, batch, seq, or the loss *value*

## 2. Where the time actually goes
Per training step, base Soup reduces over the **151,936-token vocabulary axis four times**:
- `log_softmax` (fwd) for the loss
- `log_softmax` (bwd) for the loss gradient
- `entropy` `H = -Σ p·log p`
- `accuracy` = `softmax.max`

At B=4, L=512: `4·512·151936 ≈ 311 M` floats touched **×4**, in fp32, with
**two** materialized `[B,L,V]` fp32 intermediates (~1.25 GB each).
On a T4 this is a dominant slice of wall time **and** the peak-memory hog.

## 3. What we built
Two hand-written Triton kernels:

1. **`fused_ce_h_acc_fwd`** — one fused, *online* (two-loop) streaming reduction
   over the vocab. Produces `(loss, entropy, accuracy, logsumexp, max)`.
   Runs once, keeps the running max → numerically stable, no global max pass.
2. **`fused_ce_h_acc_bwd`** — reads only `(logits, logsumexp, max)` and writes
   `dlogits = (softmax − onehot)·dloss − dH·(softmax·(logit − logsumexp) − p)`.

No full-softmax materialization in either direction.

## 4. Why the loss value is unchanged
- `loss = max + logsumexp − logits[t]` is mathematically identical to
  `F.cross_entropy`; only the fp32 reduction **order** differs (online sum).
- Verified: `|Δloss| < 1e-4`, `|Δgrad| < 1e-5` — inside the task's
  `atol=1e-3` gate and `torch.allclose`.

## 5. Integration
`solution.apply()` hooks the trainer right after `model/loss_fn` are created
and wraps `loss_fn` to use the fused kernels when `labels` are present.
Returns the exact `{"loss","acc"}` contract Soup expects.
**Zero** changes to model / optimizer / scheduler / LoRA / batch / seq.

## 6. Results (Kaggle T4, XLA)
| Metric | Base | Ours | Δ |
|---|---|---|---|
| Wall time, s | TBD | TBD | **≥ 1.10× required** |
| Peak GPU memory, MiB | TBD | TBD | **−25 % required** |
| Triton share of step | — | TBD | **≥ 5 % required** |
| Eval CE / acc | TBD | TBD | parity |

*(filled from the free-T4 run: `runs/result.json`, `runs/checks.json`)*

## 7. Checks
- ✅ custom `@triton.jit`, recognised as `OWN`, no banned libraries
- ✅ numerical parity vs PyTorch (local)
- ✅ 79/79 Soup regression tests
- ⏳ speedup ≥ 1.10×, memory −25 %, triton share ≥ 5 % → from T4 run

## 8. Limits & next steps
- Targets the **softmax reductions**, not the matmuls (~36 % of step time) —
  the next frontier (fused GEMM / attention).
- Tuned for T4 fp16; block size (16384) is not per-arch autotuned.
- Did not pursue: pad-token removal, NF4 dequant fusion, attention fusion.
