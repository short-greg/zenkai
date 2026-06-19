import torch
import torch.nn as nn

from zenkai.utils import grad_undo


class TestGradUndo(object):

    def test_grad_undo_restores_tensor_grad_to_none(self):
        t = torch.nn.parameter.Parameter(torch.randn(3))
        with grad_undo(t):
            (t.sum()).backward()
        assert t.grad is None

    def test_grad_undo_restores_previous_tensor_grad(self):
        t = torch.nn.parameter.Parameter(torch.randn(3))
        (t.sum() * 2).backward()
        original = t.grad.clone()
        with grad_undo(t):
            (t.sum() * 5).backward()
        assert torch.equal(t.grad, original)

    def test_grad_undo_restores_module_grad_to_none(self):
        module = nn.Linear(2, 3)
        with grad_undo(module):
            module(torch.randn(4, 2)).sum().backward()
        for p in module.parameters():
            assert p.grad is None

    def test_grad_undo_restores_previous_module_grad(self):
        module = nn.Linear(2, 3)
        module(torch.randn(4, 2)).sum().backward()
        originals = [p.grad.clone() for p in module.parameters()]
        with grad_undo(module):
            (module(torch.randn(4, 2)).sum() * 3).backward()
        for p, original in zip(module.parameters(), originals):
            assert torch.equal(p.grad, original)
