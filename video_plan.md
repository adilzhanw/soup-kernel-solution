# Video plan (≤ 3 min) — "What we sped up and why"

## Shot 0 — Intro (0:00–0:15)
- Title card: "Fused CE / Entropy / Accuracy — Triton kernels for soup train"
- One line: "Base Soup does 4 full-softmax passes over 151k vocab every step."

## Shot 1 — The problem (0:15–0:45)
- Show PyTorch Profiler trace of **base** run (chrome://tracing view).
- Point at the `log_softmax` / `nll_loss` / entropy / softmax-max kernels on the
  CUDA timeline — highlight that they span the full vocab axis, in fp32.
- Callout: "311M floats × 4 per step, two 1.25 GB intermediates alive at once."

## Shot 2 — Our kernels (0:45–1:30)
- Screen: `solution/fused_loss.py` — `@triton.jit fused_ce_h_acc_fwd` and `_bwd`.
- Explain in ≤3 sentences: one online streaming reduction produces loss+entropy+
  acc+logsumexp in a single pass; backward writes dlogits directly from
  (logits, logsumexp, max) — no materialized [B,L,V] softmax.
- Run `solution/test_kernels.py` → show parity output: `|Δloss| < 1e-4, |Δgrad| < 1e-5`.

## Shot 3 — The run (1:30–2:15)
- Kaggle notebook (T4 / XLA), run the whole thing.
- Show `runs/result.json` + `runs/checks.json`: wall-time speedup, peak-memory
  delta, triton share.
- Side-by-side bars: Base vs Ours (time, memory).

## Shot 4 — Result + checks (2:15–2:50)
- Table: speedup ≥ 1.10×, memory −25 %, triton ≥ 5 %, parity ✅, 10/10 checks.
- One line on what it does **not** change (loss value, model, LoRA, batch).

## Shot 5 — Close (2:50–3:00)
- "Two kernels, zero banned libs, loss value identical. Repo + notebook linked."

## Production notes
- Record the profiler trace + the two runs on the T4, then assemble with
  any screen recorder; keep total ≤ 3:00.
- Captions/subtitles on, voice-over in the participant's language is fine.
- Upload to YouTube (unlisted) or Google Drive; paste the link in SUBMISSION.md.
