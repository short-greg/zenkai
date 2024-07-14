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

zenkai.build
-------------

.. autosummary::
   :toctree: generated

   zenkai.build.BuilderFunctor
   zenkai.build.Var
   zenkai.build.Factory
   zenkai.build.BuilderArgs
   zenkai.build.Builder


zenkai.memory
-------------

.. autosummary::
   :toctree: generated

   zenkai.memory.BatchMemory


zenkai.tansaku
--------------

.. autosummary::
   :toctree: generated

   zenkai.tansaku.mean
   zenkai.tansaku.quantile
   zenkai.tansaku.median
   zenkai.tansaku.normalize
   zenkai.tansaku.NullConstraint
   zenkai.tansaku.ValueConstraint
   zenkai.tansaku.LT
   zenkai.tansaku.LTE
   zenkai.tansaku.GT
   zenkai.tansaku.GTE
   zenkai.tansaku.FuncObjective
   zenkai.tansaku.NNLinearObjective
   zenkai.tansaku.CriterionObjective
   zenkai.tansaku.PopModule
   zenkai.tansaku.PopOptimBase
   zenkai.tansaku.binary_noise
   zenkai.tansaku.add_noise
   zenkai.tansaku.cat_noise
   zenkai.tansaku.add_pop_noise
   zenkai.tansaku.cat_pop_noise
   zenkai.tansaku.NoiseReplace
   zenkai.tansaku.ExplorerNoiser
   zenkai.tansaku.Exploration
   zenkai.tansaku.RandExploration
   zenkai.tansaku.Explorer
   zenkai.tansaku.GaussianNoiser
   zenkai.tansaku.remove_noise
   zenkai.tansaku.ModuleNoise
   zenkai.tansaku.AssessmentDist
   zenkai.tansaku.EqualsAssessmentDist
   zenkai.tansaku.FreezeDropout
   zenkai.tansaku.loop_select
   zenkai.tansaku.to_pvec
   zenkai.tansaku.align_vec
   zenkai.tansaku.set_pvec
   zenkai.tansaku.acc_pvec
   zenkai.tansaku.set_gradvec
   zenkai.tansaku.acc_gradvec
   zenkai.tansaku.set_gradtvec
   zenkai.tansaku.acc_gradtvec
   zenkai.tansaku.unsqueeze_to
   zenkai.tansaku.unsqueeze_vector
   zenkai.tansaku.shape_as
   zenkai.tansaku.align
   zenkai.tansaku.separate_batch
   zenkai.tansaku.collapse_batch
   zenkai.tansaku.separate_feature
   zenkai.tansaku.collapse_feature
   zenkai.tansaku.expand_dim0
   zenkai.tansaku.flatten_dim0
   zenkai.tansaku.deflatten_dim0
   zenkai.tansaku.undo_cat1d
   zenkai.tansaku.cat1d
   zenkai.tansaku.AdaptBatch
   zenkai.tansaku.AdaptFeature
   zenkai.tansaku.TInfo
   zenkai.tansaku.best
   zenkai.tansaku.gather_selection
   zenkai.tansaku.pop_assess
   zenkai.tansaku.select_from_prob
   zenkai.tansaku.Selection
   zenkai.tansaku.Selector
   zenkai.tansaku.BestSelector
   zenkai.tansaku.TopKSelector
   zenkai.tansaku.ToProb
   zenkai.tansaku.ProbSelector
   zenkai.tansaku.ToFitnessProb
   zenkai.tansaku.ToRankProb
   zenkai.tansaku.rand_update
   zenkai.tansaku.mix_cur
   zenkai.tansaku.update_feature
   zenkai.tansaku.update_mean
   zenkai.tansaku.update_var
   zenkai.tansaku.update_momentum
   zenkai.tansaku.decay
   zenkai.tansaku.Updater
   zenkai.tansaku.calc_slope
   zenkai.tansaku.calc_scale
   zenkai.tansaku.normalize_weight
   zenkai.tansaku.softmax_weight
   zenkai.tansaku.rank_weight
   zenkai.tansaku.log_weight
   zenkai.tansaku.gauss_cdf_weight
   zenkai.tansaku.CrossOver
   zenkai.tansaku.full_crossover
   zenkai.tansaku.smooth_crossover
   zenkai.tansaku.hard_crossover
   zenkai.tansaku.gaussian_sample
   zenkai.tansaku.gaussian_noise
   zenkai.tansaku.es_dx
   zenkai.tansaku.to_gradvec
   zenkai.tansaku.cross_pairs
   zenkai.tansaku.ParentSelector
   zenkai.tansaku.binary_prob

zenkai.ensemble
---------------

.. autosummary::
   :toctree: generated

   zenkai.ensemble.weighted_votes
   zenkai.ensemble.VoteAggregator
   zenkai.ensemble.MeanVoteAggregator
   zenkai.ensemble.BinaryVoteAggregator
   zenkai.ensemble.MulticlassVoteAggregator
   zenkai.ensemble.Voter
   zenkai.ensemble.EnsembleVoter
   zenkai.ensemble.StochasticVoter


zenkai.feedback
---------------

.. autosummary::
   :toctree: generated

   zenkai.feedback.fa_target
   zenkai.feedback.FALearner
   zenkai.feedback.DFALearner
   zenkai.feedback.LinearFABuilder
   zenkai.feedback.LinearDFABuilder
   zenkai.feedback.OutT

zenkai.scikit
-------------

.. autosummary::
   :toctree: generated

   zenkai.scikit.ScikitWrapper
   zenkai.scikit.MultiOutputScikitWrapper
   zenkai.scikit.LinearBackup
   zenkai.scikit.MulticlassBackup
   zenkai.scikit.BinaryBackup

zenkai.targetprop
-----------------

.. autosummary::
   :toctree: generated

   zenkai.targetprop.LeastSquaresStepTheta
   zenkai.targetprop.LeastSquaresStepX
   zenkai.targetprop.Reversible
   zenkai.targetprop.Null
   zenkai.targetprop.TargetReverser
   zenkai.targetprop.SequenceReversible
   zenkai.targetprop.SigmoidInvertable
   zenkai.targetprop.SoftMaxReversible
   zenkai.targetprop.ReversibleMachine
   zenkai.targetprop.TPLayerLearner
   zenkai.targetprop.TPForwardLearner
   zenkai.targetprop.TPReverseLearner
   zenkai.targetprop.Rec
   zenkai.targetprop.LinearRec
   zenkai.targetprop.create_grad_target_prop
   zenkai.targetprop.BatchNorm1DReversible
   zenkai.targetprop.BoolToSigned
   zenkai.targetprop.SignedToBool
   zenkai.targetprop.DiffTPLayerLearner
   zenkai.targetprop.GradLeastSquaresLearner
   zenkai.targetprop.LeakyReLUInvertable
   zenkai.targetprop.LeastSquaresLearner
   zenkai.targetprop.LeastSquaresSolver
   zenkai.targetprop.LeastSquaresStandardSolver
   zenkai.targetprop.LeastSquaresRidgeSolver
