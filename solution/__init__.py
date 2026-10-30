"""Your solution.

The harness imports this package and calls `apply()` once, in the same process as `soup train`,
before Soup builds the model and the trainer. Patch whatever you need (transformers, peft,
bitsandbytes, Soup itself) so the training run uses your Triton kernels.

What we do
----------
Replace the causal-LM cross-entropy (a full softmax over the 151,936-token vocab on every step)
with one fused Triton kernel per direction. The kernel computes the CE *and* the per-row entropy
*and* the argmax accuracy in a single pass over the logits, and its backward is exactly
`p - onehot` (identical to `F.cross_entropy`). The trainer (TRL) then reads the fused
entropy/accuracy from a module-level slot instead of re-running two more full softmax passes.

Net effect on base Soup: 4 full-softmax reductions over the vocab per step -> 2 Triton launches,
and the fp32 `[B, L, V]` softmax temporary that the separate passes held alive is gone.
"""

import threading

from .fused_loss import FusedSoftmax

# Module-level slot holding the metrics computed by the fused CE kernel on the current step.
# One training step runs at a time on the single GPU, so a plain slot is safe.
_slot = threading.local()


def _patch_qwen2_loss_function():
    """Replace Qwen2ForCausalLM.loss_function with the fused Triton cross-entropy."""
    from transformers.models.qwen2 import modeling_qwen2
    import torch

    def loss_function(model, logits, labels, vocab_size, **kwargs):
        # The base ForCausalLMLoss shifts logits[..., :-1] vs labels[..., 1]; mirror it.
        if labels.shape == logits.shape:
            sl, slab = logits[..., :-1, :], labels[..., 1:]
        else:
            sl, slab = logits, labels
        loss_sum, entropy_sum, acc_num, acc_den = FusedSoftmax.apply(sl, slab, None)
        nbi = kwargs.get("num_items_in_batch", None)
        if nbi is None:
            nbi = (slab != -100).sum().to(torch.float32)
        loss = loss_sum / nbi.to(loss_sum.dtype)
        # Publish the fused metrics for the trainer to read (no re-computation).
        _slot.fused = (loss_sum, entropy_sum, acc_num, acc_den)
        return loss

    def _make_loss_fn(model):
        # Bind the model so `self.loss_function(logits, labels, vocab)` works without self.
        def fn(logits, labels, vocab_size, **kwargs):
            return loss_function(model, logits, labels, vocab_size, **kwargs)
        return fn

    # `loss_function` is looked up on the *instance* at call time as `self.loss_function(...)`
    # (no `self` passed). A property returning a closure bound to the instance makes it work.
    modeling_qwen2.Qwen2ForCausalLM.loss_function = property(lambda self: _make_loss_fn(self))


def _patch_trl_metrics():
    """Make TRL SFTTrainer read the fused entropy/accuracy instead of re-computing them."""
    import torch
    from trl.trainer import sft_trainer as sft_mod

    original_compute_loss = sft_mod.SFTTrainer.compute_loss

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        result = original_compute_loss(self, model, inputs, return_outputs, num_items_in_batch)
        mode = "train" if self.model.training else "eval"
        fused = getattr(_slot, "fused", None)
        if fused is None or mode not in self._metrics:
            return result
        loss_sum, entropy_sum, acc_num, acc_den = fused

        # Entropy: the base computes sum(per_row_entropy * attention_mask) / attention_mask.sum().
        # Our kernel returned the SUM over all (B, L) positions (no mask), so divide by the
        # attention-mask token count to match the base's attention-weighted mean.
        if "entropy" in self._metrics[mode]:
            if "attention_mask" in inputs:
                am = inputs["attention_mask"]
                mean_ent = (entropy_sum / am.sum().to(entropy_sum.dtype)).item()
            else:
                b, l = inputs["position_ids"].shape[:2]
                mean_ent = (entropy_sum / (b * l)).item()
            self._metrics[mode]["entropy"] = self._metrics[mode]["entropy"][:-1]
            self._metrics[mode]["entropy"].append(mean_ent)

        # Token accuracy: base appends correct/total over target tokens.
        if "mean_token_accuracy" in self._metrics[mode] and acc_den > 0:
            acc_val = (acc_num / acc_den).item()
            self._metrics[mode]["mean_token_accuracy"] = \
                self._metrics[mode]["mean_token_accuracy"][:-1]
            self._metrics[mode]["mean_token_accuracy"].append(acc_val)

        _slot.fused = None  # consume
        return result

    sft_mod.SFTTrainer.compute_loss = compute_loss


def apply():
    _patch_qwen2_loss_function()
    _patch_trl_metrics()
