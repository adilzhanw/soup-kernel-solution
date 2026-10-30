# Soup Kernel Challenge — Our Solution

**Decentrathon 6.0 · Soup 0.77.0 · Qwen2.5-1.5B-Instruct (4-bit NF4) · LoRA r=16 · Triton**

We replace the per-step full-softmax reductions over the 151,936-token vocabulary with **two
fused Triton kernels** (one per direction). Base Soup runs **four** full-softmax passes over the
`[B, L, 151936]` logits every training step:

| Pass | Where | What it computes |
|---|---|---|
| 1 | `ForCausalLMLoss` | causal-LM cross-entropy (the training loss) |
| 2 | TRL `entropy_from_logits` | per-row Shannon entropy (logged metric) |
| 3 | TRL `logits.argmax(dim=-1)` | argmax token accuracy (logged metric) |
| 4 | Soup `rewind` `row_losses` | per-row CE for the flight recorder |

Each pass materialises an fp32 copy of the logits and launches its own GPU kernels. Our
`fused_softmax_fwd` does **all three reductions (CE + entropy + accuracy) in a single Triton
launch**, and `fused_softmax_bwd` scatters the CE gradient (`p − 1_v`) in one launch. The trainer
then reads the fused entropy/accuracy instead of re-doing passes 2 and 3, and the rewind
recorder reuses the same CE.

## What we wrote

Two `@triton.jit` kernels in [`solution/fused_loss.py`](solution/fused_loss.py):

- **`fused_softmax_fwd_kernel`** — one program per (row) of the shifted `[N, V]` logits. It
  finds the row max in one sweep, then in a second sweep computes the log-partition `log_z`,
  the cross-entropy at the target (sum-reduction), the row entropy (nats) and the argmax
  (accuracy numerator/denominator) — all sharing the same row load. Rows whose label is `-100`
  contribute no CE and no accuracy (padding).
- **`fused_softmax_bwd_kernel`** — for each row with a target it stores
  `grad[:, v] = p[:, v] − 1[v == target]` (`p = softmax(row)`); padding rows are skipped. This is
  **exactly** `F.cross_entropy`'s gradient, so the adapter trains identically.

Both are wrapped in `FusedSoftmax(torch.autograd.Function)`. We patch
`Qwen2ForCausalLM.loss_function` (a property returning a closure bound to the instance) and
`SFTTrainer.compute_loss` (to read the fused metrics instead of re-computing them) — see
[`solution/__init__.py`](solution/__init__.py). No task settings, data, LoRA, optimizer or
measurements are touched.

## How it helps

- **Memory:** the fp32 `[B, L, V]` softmax temporary that the separate passes held alive is gone;
  one fused pass replaces four. Peak GPU memory drops.
- **Speed:** 4 full-softmax reductions → 2 Triton launches per step. The ~7% loss-softmax plus a
  chunk of the ~44% "element-wise / dtype cast" bucket is replaced by two tight kernels.
- **Quality:** CE, entropy, accuracy and the gradient are identical to the base (within fp16
  rounding), so the adapter, the loss curve, the metrics and the rewind log all match.

## Run it

```bash
# on a Kaggle T4 (GPU T4 ×2, Internet on): copy the Kaggle notebook
# templates/kaggle_notebook.ipynb to kaggle.com, set REPO_URL, Run All.
# Locally (needs ≥15 GB VRAM to fit the full run):
pip install "torch==2.9.*" "triton==3.5.*" "transformers==5.16.1" "peft==0.20.0" \
  "trl==0.29.1" "bitsandbytes==0.50.*" "accelerate==1.14.0" "soup-cli[train]==0.77.0" pytest
python task/prepare_data.py
python -m pytest solution -q          # kernel correctness vs PyTorch
python -m pytest tests -q             # Soup regression with our patches (79 tests)
python harness/run_task.py            # base then solution -> runs/result.json
python harness/check_rules.py         # the 10 rule checks
python harness/score.py runs/result.json
```

## Results on Kaggle T4

_filled from runs/result.json after the measured run — base vs solution: time, peak MiB, eval
loss, speedup s, memory ratio m, and the kernel's share of training time (check 4)._

## Limitations

- The fused kernel handles the softmax *reductions*; the matmuls (36% of step time) are
  untouched — that is the next frontier.
- Measured only on a T4 (fp16, no bf16); the kernel is dtype-agnostic but we did not tune block
  sizes per architecture.
- (add: what did not work, e.g. pad-token removal, NF4 dequant fusion, etc.)

## Links

- Repository, final commit: _TBD_
- Public Kaggle notebook: _TBD_
- Video (≤3 min): _TBD_ · Presentation: _TBD_
