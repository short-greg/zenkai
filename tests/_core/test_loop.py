import torch
from torch import nn

from zenkai._core import _loop


class TestMinibatchLoop:
    def test_yields_all_samples_as_minibatches(self):
        x = torch.arange(6).float().unsqueeze(1)
        batches = list(_loop.minibatch_loop(x, shuffle=False))
        assert len(batches) == 6
        collected = torch.cat([b[0] for b in batches]).view(-1)
        assert set(collected.tolist()) == set(range(6))

    def test_yields_each_tensor_for_multiple_inputs(self):
        x = torch.arange(4).float().unsqueeze(1)
        t = torch.arange(4).float().unsqueeze(1)
        batches = list(_loop.minibatch_loop(x, t, shuffle=False))
        assert len(batches) == 4
        for mb in batches:
            assert len(mb) == 2

    def test_drop_last_does_not_drop_when_divisible(self):
        x = torch.arange(3).float().unsqueeze(1)
        batches = list(_loop.minibatch_loop(x, shuffle=False, drop_last=True))
        assert len(batches) == 3


class TestModuleFilter:
    def test_yields_only_matching_submodules(self):
        module = nn.Sequential(nn.Linear(2, 2), nn.ReLU(), nn.Linear(2, 2))
        linears = list(_loop.module_filter(module, nn.Linear))
        assert len(linears) == 2
        assert all(isinstance(m, nn.Linear) for m in linears)

    def test_filters_nested_submodules_recursively(self):
        inner = nn.Sequential(nn.Linear(2, 2), nn.ReLU())
        module = nn.Sequential(inner, nn.Linear(2, 2))
        linears = list(_loop.module_filter(module, nn.Linear))
        assert len(linears) == 2


class TestModuleApply:
    def test_applies_function_to_matching_submodules(self):
        module = nn.Sequential(nn.Linear(2, 2), nn.ReLU(), nn.Linear(2, 2))
        seen = []
        _loop.module_apply(module, nn.Linear, lambda m: seen.append(m))
        assert len(seen) == 2
        assert all(isinstance(m, nn.Linear) for m in seen)

    def test_applies_to_nested_submodules_recursively(self):
        inner = nn.Sequential(nn.Linear(2, 2))
        module = nn.Sequential(inner, nn.Linear(2, 2))
        seen = []
        _loop.module_apply(module, nn.Linear, lambda m: seen.append(m))
        assert len(seen) == 2
