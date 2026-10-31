# Submission: Adilzhan Zhanali

## Links
- Repository: https://github.com/adilzhanw/soup-kernel-solution (final commit: _fill after last push_)
- Public Kaggle notebook, the saved version (same link goes into the README):
  _to be created + saved on a T4; the notebook runs `kaggle_notebook.ipynb` cell by cell_
- Demo video (up to 3 minutes): _to be linked_ (plan: `video_plan.md`)
- Presentation: `presentation.md` (to be exported to PDF/slides)

## Team (up to 4 people)
| Name | Role | GitHub | Kaggle | Contact (captain) |
|---|---|---|---|---|
| Adilzhan Zhanali | kernels + integration | adilzhanw | adilzhanzhanali | captain |

## Our kernels
| Kernel (`@triton.jit` name) | File | What it replaces | Share of the training time (check 4) |
|---|---|---|---|
| `fused_softmax_fwd_kernel` | `solution/fused_loss.py` | CE fwd + per-row entropy + argmax accuracy (3 full-vocab softmax/reduce passes, fp32) | _TBD from T4 `metrics.json` (`own_kernel_ms`)_ |
| `fused_softmax_bwd_kernel` | `solution/fused_loss.py` | CE gradient `(p − 1_v)` over 151 936 classes (fp32 re-softmax) | _TBD_ |

Ideas taken from other kernels, with links (we wrote the code ourselves):
- Fused softmax cross-entropy is a standard technique (cf. liger-kernel / Unsloth fused CE);
  we implemented it ourselves in Triton, without importing any of those libraries.

## Results on Kaggle T4 (from `runs/result.json` and `runs/checks.json` of the notebook above)
| | Base Soup | Ours | Ratio |
|---|---|---|---|
| Time of `soup train`, s | _TBD_ | _TBD_ | speedup x (need ≥ 1.10) |
| Peak GPU memory, MiB | _TBD_ | _TBD_ | base/ours x (need ≤ 0.75) |
| Eval loss (public rows) | _TBD_ | _TBD_ | % vs base (need ≤ +0.5 %) |

Local (RTX 5070, 8 GB) unit verification already done:
- `pytest solution`: **7/7 passed** — loss / grad / entropy / acc vs `F.cross_entropy`,
  fp16 + fp32, full task shapes `[4, 512, 151936]`, with `-100` masks.
- Loss rel. diff `< 8e-7`, max gradient diff `< 2.4e-7`; both kernels detected and timed by
  the harness `TritonWatch` as **OWN** (`triton_files: solution/fused_loss.py`), zero banned
  modules loaded.
- The 8 GB laptop GPU cannot run the fixed task end-to-end (the *base* path OOMs on step 2),
  so wall-time / memory / share numbers are T4-only.

## Rules self-check
- [ ] All 10 checks of `harness/check_rules.py` passed (T4 run pending; static parts pass)
- [x] README.md rewritten for our solution: what we did, how to run it, results, limits
- [x] `harness/` and `task/` unmodified (byte-identical to the starter)
- [x] No banned libraries (`unsloth`, `liger_kernel`, `xformers`, `flash_attn`, `torchao`,
      `apex`, `transformer_engine`, `triton.ops`, prebuilt kernel repos) — static scan + runtime
      `banned_modules_loaded: []`
- [x] No hyperparameter/model/data changes

## Presentation covers
Which operations we sped up and why, how the kernels work (online two-loop max+sum-exp,
exact `p − 1_v` backward), results against the base (speed, memory, quality), limits and
next steps.

## Limitations
- The fused kernel targets the softmax *reductions* (loss + Soup metrics), not the matmuls —
  that is the next frontier.
- Tuned for a T4 (fp16); block sizes (BLOCK=16384) are not per-arch autotuned.
- The metrics hook replaces Soup's separate entropy/accuracy passes with the fused kernel's
  outputs; if a future Soup version changes `SFTTrainer.compute_loss`'s contract, the
  monkey-patch in `solution/__init__.py` needs a one-line follow-up.
