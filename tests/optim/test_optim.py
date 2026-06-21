import pytest
import torch

from zenkai.optimz._optim import PopOptimBase


class DummyPopOptim(PopOptimBase):
    """Concrete PopOptimBase used to exercise the base behaviour."""

    def __init__(self, decay: float = None):
        super().__init__(decay)
        self.stepped = False

    def step(self):
        self.stepped = True


class TestPopOptimBase:

    def test_cannot_instantiate_abstract_base(self):
        with pytest.raises(TypeError):
            PopOptimBase()

    def test_init_sets_decay_and_none_assessment(self):
        optim = DummyPopOptim(decay=0.5)
        assert optim.decay == 0.5
        assert optim.assessment is None

    def test_accumulate_assessment_doubles_first_assessment_without_decay(self):
        # On the first call assessment is set to ``a`` then the no-decay branch
        # adds ``a`` again, so the result is ``2 * a``.
        optim = DummyPopOptim()
        a = torch.tensor([1.0, 2.0])
        optim.accumulate_assessment(a)
        assert torch.equal(optim.assessment, 2 * a)

    def test_accumulate_assessment_adds_without_decay(self):
        optim = DummyPopOptim()
        optim.accumulate_assessment(torch.tensor([1.0, 2.0]))  # -> [2, 4]
        optim.accumulate_assessment(torch.tensor([3.0, 4.0]))  # -> [5, 8]
        assert torch.equal(optim.assessment, torch.tensor([5.0, 8.0]))

    def test_accumulate_assessment_applies_decay(self):
        optim = DummyPopOptim(decay=0.5)
        # First call: assessment=[2,2], then [2,2] + 0.5*[2,2] = [3,3].
        optim.accumulate_assessment(torch.tensor([2.0, 2.0]))
        # Second call: [1,1] + 0.5*[3,3] = [2.5, 2.5].
        optim.accumulate_assessment(torch.tensor([1.0, 1.0]))
        assert torch.equal(optim.assessment, torch.tensor([2.5, 2.5]))

    def test_zero_assessment_resets_to_none(self):
        optim = DummyPopOptim()
        optim.accumulate_assessment(torch.tensor([1.0]))
        optim.zero_assessment()
        assert optim.assessment is None

    def test_step_is_callable_on_concrete_subclass(self):
        optim = DummyPopOptim()
        optim.step()
        assert optim.stepped is True
