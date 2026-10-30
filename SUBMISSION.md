# Submission: adk / Adilzhan Zhanali

## Links
- Repository, final commit: https://github.com/adilzhanw/soup-kernel-solution/commit/6da837d2eba4465bbd5e2364126b9cb14befa20b
- Public Kaggle notebook, the saved version (`COMMIT` in it = the sha above; the same link must
  be in your README):
  https://www.kaggle.com/code/adilzhanw/soup-kernel-benchmark  _(to be created + saved; COMMIT cell pins the sha above)_
- Demo video (up to 3 minutes): _(to be linked)_
- Presentation: _(to be linked)_

## Team (up to 4 people)
| Name | Role | GitHub | Kaggle | Contact (captain) |
|---|---|---|---|---|
| Adilzhan Zhanali | kernels + integration | adilzhanw | adilzhanw | captain |

## Our kernels
| Kernel (`@triton.jit` name) | File | What it replaces | Share of the training time (check 4) |
|---|---|---|---|
| `fused_ce_h_acc_fwd` | `solution/fused_loss.py` | CE + entropy + argmax-accuracy (3 full-softmax passes) | % |
| `fused_ce_h_acc_bwd` | `solution/fused_loss.py` | CE gradient scatter (`p − 1_v`) | % |

Ideas taken from other kernels, with links (we wrote the code ourselves):
- Fused softmax cross-entropy is a standard technique (cf. liger-kernel / Unsloth fused CE); we
  implemented it ourselves in Triton, without importing any of those libraries.

## Results on Kaggle T4 (from `runs/result.json` and `runs/checks.json` of the notebook above)
| | Base Soup | Ours | Ratio |
|---|---|---|---|
| Time of `soup train`, s | | | speedup x |
| Peak GPU memory, MiB | | | base/ours x |
| Eval loss (public rows) | | | % vs base |

- [ ] All 10 checks of `harness/check_rules.py` passed (if not: which failed and why)
- [ ] README.md rewritten for our solution: what we did, how to run it, results, limits

## Presentation covers
Which operations you sped up and why, how the kernels work, results against the base (speed,
memory, quality), limits and next steps.

## Limitations
- The fused kernel targets the softmax *reductions* (loss + metrics + rewind), not the matmuls
  (~36% of step time) — that is the next frontier.
- Tuned for a T4 (fp16); block sizes (BLOCK=16384) are not per-arch autotuned.
- (add: what did not work — pad-token removal, NF4 dequant fusion, attention fusion, etc.)
