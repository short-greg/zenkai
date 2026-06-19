import pytest
import torch.nn as nn

from zenkai.utils import checkattr, module_factory


class TestModuleFactory(object):

    def test_module_factory_creates_module_from_string(self):
        module = module_factory("Linear", 2, 3)
        assert isinstance(module, nn.Linear)
        assert module.in_features == 2
        assert module.out_features == 3

    def test_module_factory_passes_through_existing_module(self):
        linear = nn.Linear(2, 3)
        assert module_factory(linear) is linear

    def test_module_factory_raises_when_args_given_with_module(self):
        linear = nn.Linear(2, 3)
        with pytest.raises(ValueError):
            module_factory(linear, 4)

    def test_module_factory_raises_when_kwargs_given_with_module(self):
        linear = nn.Linear(2, 3)
        with pytest.raises(ValueError):
            module_factory(linear, out_features=4)


class TestCheckAttr(object):

    def test_checkattr_allows_call_when_attribute_present(self):

        class Holder(object):
            x = 1

            @checkattr("x")
            def get(self):
                return self.x

        assert Holder().get() == 1

    def test_checkattr_raises_when_attribute_missing(self):

        class Holder(object):

            @checkattr("x")
            def get(self):
                return self.x

        with pytest.raises(AttributeError):
            Holder().get()
