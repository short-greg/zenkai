import torch

from sklearn import linear_model
import numpy as np

# from zenkai.scikit import _scikit_mod
from zenkai.scikit import (
    ScikitRegressor, ScikitBinary, ScikitMulticlass
)

class TestScikitRegressor:
    def test_scikit_regressor_outputs_correct_size(self):

        torch.manual_seed(1)
        np.random.seed(1)
        regressor = linear_model.SGDRegressor()
        training_data = np.random.randn(4, 2).astype(np.float32), np.random.randn(
            4
        ).astype(np.float32)
        regressor.fit(training_data[0], training_data[1])

        regressor_wrapper = ScikitRegressor(
            regressor, 2, None
        )
        regressor_wrapper._fitted = True
        test_data = np.random.randn(4, 2).astype(np.float32), np.random.randn(4).astype(
            np.float32
        )

        y = regressor_wrapper(torch.from_numpy(test_data[0]))
        t = torch.from_numpy(regressor.predict(test_data[0]).astype(np.float32)).unsqueeze(1)
        assert torch.isclose(y, t).all()

    def test_scikit_regressor_outputs_correct_size_for_multiout(self):

        torch.manual_seed(1)
        np.random.seed(1)
        regressor = linear_model.SGDRegressor()
        training_data = torch.randn(4, 2), torch.randn(4, 1)

        regressor_wrapper = ScikitRegressor.multi(regressor, 2, 1)

        regressor_wrapper.fit(
            training_data[0], training_data[1])

        test_data = np.random.randn(4, 2).astype(np.float32), np.random.randn(
            4, 1
        ).astype(np.float32)

        y = regressor_wrapper(torch.from_numpy(test_data[0]))
        t = torch.from_numpy(
            regressor_wrapper.models[0]._estimator.predict(test_data[0]).astype(np.float32)
        )[:,None]
        assert torch.isclose(y, t, 1e-4).all()

    def test_scikit_regressor_outputs_correct_size_for_baseline(self):

        torch.manual_seed(1)
        np.random.seed(1)
        regressor = linear_model.SGDRegressor()

        regressor_wrapper = ScikitRegressor(
            regressor, 2, None
        )
        test_data = np.random.randn(4, 2).astype(
            np.float32
        ), np.random.randn(4).astype(
            np.float32
        )

        y = regressor_wrapper(
            torch.from_numpy(test_data[0])
        )
        assert y.shape == torch.Size([4, 1])

#     def test_scikit_regressor_outputs_correct_size_for_baseline_with_multioutput(self):

#         torch.manual_seed(1)
#         np.random.seed(1)
#         regressor = linear_model.SGDRegressor()

#         regressor_wrapper = _scikit_mod.MultiOutputScikitWrapper.regressor(regressor, 2, 3)
#         test_data = np.random.randn(4, 2).astype(np.float32), np.random.randn(
#             4, 3
#         ).astype(np.float32)

#         y = regressor_wrapper(torch.from_numpy(test_data[0]))
#         assert y.shape == torch.Size([4, 3])


class TestScikitWrapperWithClass:

    def test_scikit_multiclass_outputs_same_results(self):

        torch.manual_seed(1)
        np.random.seed(1)
        classifier = linear_model.SGDClassifier()
        training_data = np.random.randn(4, 4).astype(np.float32), np.random.randint(
            0, 4, (4,)
        ).astype(np.float32)
        classifier.fit(training_data[0], training_data[1])

        classifier_wrapper = ScikitMulticlass(classifier, 2, 4)
        classifier_wrapper._fitted = True
        test_data = np.random.randn(4, 4).astype(np.float32), np.random.randint(
            0, 4, (4,)
        ).astype(np.float32)

        y = classifier_wrapper(torch.from_numpy(test_data[0]))
        t = torch.from_numpy(
            classifier.predict(test_data[0]).astype(np.int64)
        ).unsqueeze(-1)
        assert torch.isclose(y, t).all()

    def test_scikit_classifier_outputs_correct_size_for_multiout(self):

        torch.manual_seed(1)
        np.random.seed(1)
        classifier = linear_model.LogisticRegression()
        training_data = torch.randn(16, 4), torch.randint(0, 4, (16, 2))
        classifier_wrapper = ScikitMulticlass.multi(
            classifier, 2, 2, 4
        )

        classifier_wrapper.fit(
            training_data[0], training_data[1]
        )

        test_data = np.random.randn(4, 4).astype(np.float32), np.random.randint(
            0, 4, (4, 2)
        ).astype(np.float32)
        y = classifier_wrapper(torch.from_numpy(test_data[0]))
        assert y.shape == torch.Size([4, 2])

    def test_scikit_classifier_outputs_correct_size_for_baseline(self):

        torch.manual_seed(1)
        np.random.seed(1)
        classifier = linear_model.LogisticRegression()

        classifier_wrapper = ScikitMulticlass.multi(
            classifier, 2, 4, 2
        )
        test_data = np.random.randn(4, 2).astype(np.float32), np.random.randn(4).astype(
            np.float32
        )

        y = classifier_wrapper(torch.from_numpy(test_data[0]))
        assert y.shape == torch.Size([4, 4])

    def test_scikit_classifier_outputs_correct_size_for_baseline_with_multioutput(self):

        torch.manual_seed(1)
        np.random.seed(1)
        classifier = linear_model.SGDClassifier()
        classifier_wrapper = ScikitMulticlass.multi(
            classifier, 2, 4, 2
        )

        test_data = np.random.randn(4, 2).astype(np.float32), np.random.randn(
            4, 1
        ).astype(np.float32)

        y = classifier_wrapper(torch.from_numpy(test_data[0]))
        assert y.shape == torch.Size([4, 4])

    def test_scikit_binary_outputs_correct_size_for_baseline_with_multioutput(self):

        torch.manual_seed(1)
        np.random.seed(1)
        classifier = linear_model.SGDClassifier()
        classifier_wrapper = ScikitBinary.multi(classifier, 2, 4)

        test_data = np.random.randn(4, 2).astype(np.float32), np.random.randn(
            4, 1
        ).astype(np.float32)

        y = classifier_wrapper(torch.from_numpy(test_data[0]))
        assert y.shape == torch.Size([4, 4])

    def test_scikit_binary_outputs_correct_size_for_baseline_with_multioutput_after_training(
        self,
    ):

        torch.manual_seed(1)
        np.random.seed(1)
        classifier = linear_model.LogisticRegression()
        classifier_wrapper = ScikitBinary.multi(classifier, 2, 4)

        training_data = torch.randn(4, 2), torch.randint(0, 4, (4, 4))
        classifier_wrapper.fit(training_data[0], training_data[1])
        test_data = np.random.randn(4, 2).astype(np.float32), np.random.randn(
            4, 4
        ).astype(np.float32)

        y = classifier_wrapper(torch.from_numpy(test_data[0]))
        assert y.shape == torch.Size([4, 4])
