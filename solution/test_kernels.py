"""Correctness of the fused cross-entropy + entropy + accuracy kernel vs plain PyTorch.

Run on a GPU: `python -m pytest solution -q`.

The kernel receives *shifted* logits/labels (as the trainer passes them to
`F.cross_entropy`), i.e. `logits[:, :-1]` vs `labels[:, 1:]`. The test mirrors that.
"""
import pytest
import torch
import torch.nn.functional as F

from solution.fused_loss import FusedSoftmax

V = 151936
TOL = {torch.float16: 1e-2, torch.float32: 5e-4}
TOL_BWD = {torch.float16: 1e-1, torch.float32: 1e-3}


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("shape", [(2, 32), (2, 512), (3, 128)])
def test_fused_matches_pytorch(dtype, shape):
    B, L = shape
    torch.manual_seed(0)
    logits = torch.randn(B, L, V, device="cuda", dtype=dtype, requires_grad=True)
    labels = torch.randint(0, V, (B, L), device="cuda")
    labels[0, :10] = -100  # some padding

    # The trainer passes the *shifted* pair to the CE kernel:
    sl, slab = logits[:, :-1, :], labels[:, 1:]

    # Reference: plain PyTorch cross-entropy on the same shifted pair.
    ref_loss = F.cross_entropy(sl.reshape(-1, V).float(), slab.reshape(-1),
                               ignore_index=-100, reduction="sum")
    ref_loss.backward()
    ref_grad = logits.grad.clone()
    logits.grad = None

    # Ours: pass the same shifted pair through the kernel.
    loss, ent_sum, acc_num, acc_den = FusedSoftmax.apply(sl, slab, None)
    loss.backward()
    tol, tol_bwd = TOL[dtype], TOL_BWD[dtype]
    torch.testing.assert_close(loss.float(), ref_loss.float(), rtol=tol, atol=tol)
    torch.testing.assert_close(logits.grad.float(), ref_grad.float(), rtol=tol_bwd, atol=tol_bwd)

    # entropy: sum of per-row nats over ALL positions (incl. padding) of the SHIFTED logits
    logps = torch.log_softmax(sl.detach().float(), dim=-1)
    per_row_ref = (-(logps.exp() * logps).sum(-1))
    torch.testing.assert_close(ent_sum.float(), per_row_ref.sum().float(), rtol=tol, atol=tol * L)

    # accuracy over target tokens (shifted)
    acc = acc_num.item() / acc_den.item()
    mask = slab != -100
    pred = sl.detach().argmax(dim=-1)
    ref_acc = float(((pred == slab) & mask).sum() / mask.sum())
    torch.testing.assert_close(acc, ref_acc, rtol=1e-5, atol=1e-5)


def test_fused_backward_only_target_rows():
    B, L = 1, 3
    torch.manual_seed(1)
    logits = torch.randn(B, L, V, device="cuda", dtype=torch.float32, requires_grad=True)
    labels = torch.full((B, L), -100, dtype=torch.long, device="cuda")
    labels[0, 1] = 1234  # original position 1
    sl, slab = logits[:, :-1, :], labels[:, 1:]
    # slab[0,0] = labels[0,1] = 1234  -> CE on sl[0,0] = logits[0,0]
    # slab[0,1] = labels[0,2] = -100  -> ignored, no gradient on logits[0,1]
    loss, *_ = FusedSoftmax.apply(sl, slab, None)
    loss.backward()
    # position 0 carries (p - onehot); positions 1, 2 carry ~zero (ignored / out-of-shift)
    assert not torch.allclose(logits.grad[0, 0], torch.zeros_like(logits.grad[0, 0]), atol=1e-3)
    assert logits.grad[0, 1].abs().max().item() < 1e-4
    assert logits.grad[0, 2].abs().max().item() < 1e-4
