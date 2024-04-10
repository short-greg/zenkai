# # 3rd party
# import torch

# # local
# from zenkai import Assessment
# from zenkai import kaku, Population
# from zenkai.tansaku import _particle


# class TestGaussianParticleUpdater:

#     def test_update_updates_the_mean_and_var(self):

#         population = Population(x=torch.randn(4, 3, 2), t=torch.randn(4, 4, 2))
#         population.report(Assessment(torch.rand(4)))
#         updater = _particle.GaussianParticleUpdater()
#         updater.update(population)
#         assert updater.mean is not None
#         assert updater.var is None

#     def test_update_updates_the_mean_after_two(self):

#         population = Population(x=torch.randn(4, 3, 2), t=torch.randn(4, 4, 2))
#         population2 = Population(x=torch.randn(4, 3, 2), t=torch.randn(4, 4, 2))
#         population.report(Assessment(torch.rand(4)))
#         population2.report(Assessment(torch.rand(4)))
#         updater = _particle.GaussianParticleUpdater()
#         updater.update(population)
#         mean_before = updater.mean
#         # var_before = updater.var
#         updater.update(population2)
#         assert (updater.mean != mean_before).any()
#         # assert (updater.var != var_before).any()

#     def test_update_updates_the_var_after_three(self):

#         population = Population(x=torch.randn(4, 3, 2), t=torch.randn(4, 4, 2))
#         population2 = Population(x=torch.randn(4, 3, 2), t=torch.randn(4, 4, 2))
#         population3 = Population(x=torch.randn(4, 3, 2), t=torch.randn(4, 4, 2))
#         population.report(Assessment(torch.rand(4)))
#         population2.report(Assessment(torch.rand(4)))
#         population3.report(Assessment(torch.rand(4)))
#         updater = _particle.GaussianParticleUpdater()
#         updater.update(population)
#         updater.update(population2)
#         var_before = updater.var
#         updater.update(population2)
#         assert (updater.var != var_before).any()
#         # assert (updater.var != var_before).any()

#     def test_call_outputs_a_weighted_version(self):

#         population = Population(x=torch.randn(4, 3, 2), t=torch.randn(4, 4, 2))
#         population.report(Assessment(torch.rand(4)))
#         updater = _particle.GaussianParticleUpdater()
#         updater.update(population)
#         particle = updater(population)
#         assert isinstance(particle, Population)

#     def test_call_outputs_a_weighted_version_after_three_updates(self):

#         population = Population(x=torch.randn(4, 3, 2), t=torch.randn(4, 4, 2))
#         population2 = Population(x=torch.randn(4, 3, 2), t=torch.randn(4, 4, 2))
#         population3 = Population(x=torch.randn(4, 3, 2), t=torch.randn(4, 4, 2))
#         population.report(Assessment(torch.rand(4)))
#         population2.report(Assessment(torch.rand(4)))
#         population3.report(Assessment(torch.rand(4)))
#         updater = _particle.GaussianParticleUpdater()
#         updater.update(population)
#         updater(population2)
#         updater.update(population2)
#         updater(population2)
#         updater.update(population3)
#         particle = updater(population3)
#         assert particle['x'].shape == population['x'].shape
#         assert particle['t'].shape == population['t'].shape


#     def test_call_outputs_a_weighted_version_after_three_updates_with_multi_dim_assessment(self):

#         population = Population(x=torch.randn(4, 3, 2), t=torch.randn(4, 3, 4))
#         population2 = Population(x=torch.randn(4, 3, 2), t=torch.randn(4, 3, 4))
#         population3 = Population(x=torch.randn(4, 3, 2), t=torch.randn(4, 3, 4))
#         population.report(Assessment(torch.rand(4, 3)))
#         population2.report(Assessment(torch.rand(4, 3)))
#         population3.report(Assessment(torch.rand(4, 3)))
#         updater = _particle.GaussianParticleUpdater()
#         updater.update(population)
#         updater(population2)
#         updater.update(population2)
#         updater(population2)
#         updater.update(population3)
#         particle = updater(population3)
#         assert particle['x'].shape == population['x'].shape
#         assert particle['t'].shape == population['t'].shape


# class TestGlobalBest:

#     def test_global_particle_smooth_outputs_individual(self):

#         torch.manual_seed(1)

#         global_best = _particle.GlobalBest(_particle.GaussianParticleUpdater())

#         population = Population(x=torch.randn(4, 3, 2), t=torch.randn(4, 3, 4))
#         population2 = Population(x=torch.randn(4, 3, 2), t=torch.randn(4, 3, 4))
#         population3 = Population(x=torch.randn(4, 3, 2), t=torch.randn(4, 3, 4))
#         population.report(Assessment(torch.rand(4, 3)))
#         population2.report(Assessment(torch.rand(4, 3)))
#         population3.report(Assessment(torch.rand(4, 3)))

#         best1 = global_best(population)
#         best2 = global_best(population2)

#         assert (best1['x'] != best2['x']).any()


# class TestLocalBest:

#     def test_local_best_outputs_correct_population_size(self):

#         torch.manual_seed(1)

#         local_best = _particle.LocalBest(_particle.GaussianParticleUpdater())
#         population = Population(x=torch.randn(4, 3, 2), t=torch.randn(4, 3, 4))
#         population2 = Population(x=torch.randn(4, 3, 2), t=torch.randn(4, 3, 4))
#         population3 = Population(x=torch.randn(4, 3, 2), t=torch.randn(4, 3, 4))
#         population.report(Assessment(torch.rand(4, 3)))
#         population2.report(Assessment(torch.rand(4, 3)))
#         population3.report(Assessment(torch.rand(4, 3)))
#         best1 = local_best(population)
#         best2 = local_best(population2)
#         assert (best1['x'].shape == population['x'].shape)
#         assert (best1['x'] != best2['x']).any()
