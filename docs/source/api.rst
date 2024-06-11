.. _api:


API Reference
=============

zenkai (kaku)
-------------

.. autosummary::
   :toctree: generated

   zenkai.LearningMachine
   zenkai.GradLearner
   zenkai.IO
   zenkai.Idx
   zenkai.Reduction
   zenkai.Criterion
   zenkai.XCriterion
   zenkai.CompositeXCriterion
   zenkai.CompositeCriterion
   zenkai.NNLoss
   zenkai.AssessmentLog
   zenkai.zip_assess
   zenkai.GradStepTheta
   zenkai.GradStepX
   zenkai.GradLearner
   zenkai.GradIdxLearner
   zenkai.IdxLoop
   zenkai.IterStepTheta
   zenkai.FeatureLimitGen
   zenkai.RandomFeatureIdxGen
   zenkai.LayerAssessor
   zenkai.StepXLayerAssessor
   zenkai.StepFullLayerAssessor
   zenkai.acc_dep
   zenkai.step_dep
   zenkai.forward_dep
   zenkai.LMode
   zenkai.set_lmode
   zenkai.StepXHook
   zenkai.StepHook
   zenkai.ForwardHook
   zenkai.LearnerPostHook
   zenkai.StepX
   zenkai.StepTheta
   zenkai.BatchIdxStepTheta
   zenkai.FeatureIdxStepTheta
   zenkai.BatchIdxStepX
   zenkai.FeatureIdxStepX
   zenkai.OutDepStepTheta
   zenkai.InDepStepX
   zenkai.State
   zenkai.NullStepTheta
   zenkai.NullStepX
   zenkai.NullLearner
   zenkai.Objective
   zenkai.Constraint
   zenkai.CompoundConstraint
   zenkai.impose
   zenkai.NullOptim
   zenkai.OptimFactory
   zenkai.CompOptim
   zenkai.ParamFilter
   zenkai.Fit


zenkai.params
-------------

.. autosummary::
   :toctree: generated

   zenkai.params.get_p
   zenkai.params.get_grad
   zenkai.params.to_pvec
   zenkai.params.to_gradvec
   zenkai.params.align_vec
   zenkai.params.set_pvec
   zenkai.params.acc_pvec
   zenkai.params.set_gradvec
   zenkai.params.acc_gradvec
   zenkai.params.set_gradtvec
   zenkai.params.acc_gradtvec
   zenkai.params.get_params
   zenkai.params.to_df
   zenkai.params.to_series
   zenkai.params.get_multp
   zenkai.params.loop_p
   zenkai.params.apply_p
   zenkai.params.apply_grad
   zenkai.params.set_params
   zenkai.params.acc_params
   zenkai.params.set_grad
   zenkai.params.set_gradt
   zenkai.params.acc_grad
   zenkai.params.acc_gradt
   zenkai.params.update_model_params
   zenkai.params.reg_p
   zenkai.params.undo_grad
