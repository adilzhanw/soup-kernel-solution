"""Fused causal-LM cross-entropy + row entropy + argmax accuracy, forward and backward, in Triton.

Why this kernel exists
----------------------
Base Soup (TRL SFT) computes, on every step, after the model returns the full
``[B, L, V]`` logits (``V = 151,936`` for Qwen2.5-1.5B):

1. the causal-LM cross-entropy (``ForCausalLMLoss``) — a full softmax over the vocab,
2. the per-row Shannon entropy (``entropy_from_logits``, fp32, chunked) — a second full softmax,
3. the argmax token accuracy (``logits.argmax(dim=-1)``) — a third full pass over the vocab,
4. Soup's ``rewind`` flight recorder — a fourth pass (per-row CE via ``row_losses``).

Four reductions over the same big tensor, each keeping an fp32 copy of the logits alive, plus
the CE gradient scatter in the backward. On the T4 this is the loss's ~7% plus a chunk of the
~44% "element-wise / dtype cast" bucket, and it dominates the logits' activation memory.

This file replaces the softmax *reductions* with one fused Triton kernel per direction:

* forward (``fused_softmax_fwd``): one program per (row) reads the row in fp32, finds the running
  max, then in one more sweep computes the log-partition ``log_z``, the cross-entropy at the
  target (sum-reduction, so the trainer divides by ``num_items_in_batch`` exactly as the base),
  the row entropy (nats), and the argmax (accuracy numerator/denominator).
* backward (``fused_softmax_bwd``): for each row with a target it stores
  ``grad[:, v] = p[:, v] - 1[v == target]`` (``p = softmax(row)``); rows whose label is ``-100``
  are skipped (they carry no gradient). This is exactly ``F.cross_entropy``'s gradient.

The wrapper is a ``torch.autograd.Function``. It is **not** the example's kernel, not Inductor's,
and it is not a ready-made kernel library: the whole thing is our code.
"""

import torch
import triton
import triton.language as tl

NEG_INF = tl.constexpr(-1e30)  # large negative; must be a constexpr to be visible inside @jit
# (not -3.4e38: subtracting it from a normal logit overflows fp32 -> inf -> nan)
_BLOCK = 16384  # > V so a single program covers the whole row (Qwen vocab 151,936)


@triton.jit
def fused_softmax_fwd_kernel(
    logits_ptr, labels_ptr,
    loss_sum_ptr, entropy_ptr, acc_num_ptr, acc_den_ptr,
    V,
    IGNORE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    base = row.to(tl.int64) * V
    cols = tl.arange(0, BLOCK)
    target = tl.load(labels_ptr + row)

    # Pass 1: stable row max (masked so the final partial block reads no OOB).
    max_val = NEG_INF
    for start in range(0, V, BLOCK):
        c = start + cols
        x = tl.load(logits_ptr + base + c, mask=c < V, other=NEG_INF).to(tl.float32)
        max_val = tl.maximum(max_val, tl.max(x, axis=0))

    # Pass 2: with the max fixed, one sweep computes exp_sum, the target value, the argmax and
    # the entropy partial (sum p * (x - max)). All four share the same row load.
    exp_sum = 0.0
    best = NEG_INF
    target_val = 0.0
    ent = 0.0
    for start in range(0, V, BLOCK):
        c = start + cols
        m = c < V
        x = tl.load(logits_ptr + base + c, mask=m, other=NEG_INF).to(tl.float32)
        e = tl.exp(x - max_val)
        e = tl.where(m, e, 0.0)
        exp_sum += tl.sum(e, axis=0)
        row_max = tl.max(tl.where(m, x, NEG_INF), axis=0)
        if row_max > best:
            best = row_max
        if tl.sum((c == target).to(tl.int32), axis=0) > 0:
            target_val = tl.sum(tl.where((c == target) & m, x, 0.0), axis=0)
        ent += tl.sum(tl.where(m, e * (x - max_val), 0.0), axis=0)

    log_z = max_val + tl.log(exp_sum)
    # H (nats) = -sum p log p = log z - sum p x.  sum p x = E[x] = (sum e * (x - max)) / exp_sum.
    e_mean = ent / exp_sum          # E[x - max]
    ex_mean = e_mean + max_val      # E[x]
    row_entropy = log_z - ex_mean

    if target != IGNORE:
        tl.atomic_add(loss_sum_ptr, log_z - target_val)          # CE = log_z - x_target
        tl.atomic_add(acc_num_ptr, 1 if best == target_val else 0)
        tl.atomic_add(acc_den_ptr, 1)
    tl.atomic_add(entropy_ptr, row_entropy)


@triton.jit
def fused_softmax_bwd_kernel(
    logits_ptr, labels_ptr, grad_ptr,
    V, stride,
    IGNORE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    gbase = row.to(tl.int64) * stride
    lbase = row.to(tl.int64) * V
    cols = tl.arange(0, BLOCK)
    target = tl.load(labels_ptr + row)
    if target == IGNORE:
        return

    max_val = NEG_INF
    for start in range(0, V, BLOCK):
        c = start + cols
        m = c < V
        x = tl.load(logits_ptr + lbase + c, mask=m, other=NEG_INF).to(tl.float32)
        max_val = tl.maximum(max_val, tl.max(x, axis=0))
    exp_sum = 0.0
    for start in range(0, V, BLOCK):
        c = start + cols
        m = c < V
        x = tl.load(logits_ptr + lbase + c, mask=m, other=NEG_INF).to(tl.float32)
        exp_sum += tl.sum(tl.where(m, tl.exp(x - max_val), 0.0), axis=0)
    for start in range(0, V, BLOCK):
        c = start + cols
        m = c < V
        x = tl.load(logits_ptr + lbase + c, mask=m, other=NEG_INF).to(tl.float32)
        p = tl.where(m, tl.exp(x - max_val) / exp_sum, 0.0)
        g = p - (c == target).to(tl.float32)
        tl.store(grad_ptr + gbase + c, g.to(grad_ptr.dtype.element_ty), mask=m)


class FusedSoftmax(torch.autograd.Function):
    """Drop-in for ``F.cross_entropy`` (sum reduction) that also returns entropy/accuracy.

    Returns ``(loss, entropy_sum, acc_num, acc_den)`` where:
      * ``loss``        = sum of per-token CE over tokens with a target (the trainer divides by
        ``num_items_in_batch`` exactly as the base does),
      * ``entropy_sum`` = sum of the per-row nats entropy over ALL positions (incl. padding),
      * ``acc_num`` / ``acc_den`` = argmax-correct / target-token counts over target tokens.

    Callers pass the *shifted* logits/labels (``logits[:, :-1]`` vs ``labels[:, 1:]``).
    """

    @staticmethod
    def forward(ctx, logits, labels, num_items_in_batch=None):
        orig_shape = logits.shape
        if logits.dim() == 3:
            b, l, v = logits.shape
            logits2d = logits.reshape(-1, v)
        else:
            b, l, v = 1, logits.size(0), logits.size(-1)
            logits2d = logits
        n = logits2d.size(0)
        labels = labels.reshape(-1).contiguous()
        logits2d = logits2d.contiguous()

        loss_sum = torch.zeros((), device=logits.device, dtype=torch.float32)
        entropy_sum = torch.zeros((), device=logits.device, dtype=torch.float32)
        acc_num = torch.zeros((), device=logits.device, dtype=torch.float32)
        acc_den = torch.zeros((), device=logits.device, dtype=torch.float32)

        fused_softmax_fwd_kernel[(n,)](
            logits2d, labels, loss_sum, entropy_sum, acc_num, acc_den,
            v, IGNORE=-100, BLOCK=_BLOCK, num_warps=8,
        )
        ctx.save_for_backward(logits2d, labels)
        ctx.orig_shape = orig_shape
        ctx.n = n
        ctx.v = v
        return loss_sum, entropy_sum, acc_num, acc_den

    @staticmethod
    def backward(ctx, dloss, dentropy, daccnum, daccden):
        logits2d, labels = ctx.saved_tensors
        n, v = ctx.n, ctx.v
        # Zero-init: rows whose label is IGNORE must keep a zero gradient (CE ignores them).
        pad = (16384 - (v % 16384)) % 16384
        stride = v + pad
        grad = torch.zeros((n, stride), device=logits2d.device, dtype=torch.float32)
        fused_softmax_bwd_kernel[(n,)](
            logits2d, labels, grad, v, stride, IGNORE=-100, BLOCK=16384, num_warps=8,
        )
        grad = grad[:, :v].view(ctx.orig_shape)
        return grad * dloss.to(grad.dtype), None, None


def fused_cross_entropy(logits, labels, num_items_in_batch=None):
    """Convenience wrapper: returns ``(loss, entropy_sum, acc_num, acc_den)``."""
    return FusedSoftmax.apply(logits, labels, num_items_in_batch)
