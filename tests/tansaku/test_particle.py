# 3rd party
import torch

# local
from zenkai import Assessment
from zenkai import kaku
from zenkai.tansaku import _particle


# class TestGlobalParticleSmooth:

#     def test_global_particle_smooth_outputs_individual(self):

#         torch.manual_seed(1)

#         state = kaku.State()
#         particle = _particle.GlobalParticleSmooth(0)
#         population1 = kaku.Population(x=torch.rand(4, 4, 2))
#         population1.report(Assessment(torch.tensor([0.1, 0.2, 0.8, 1.0])))
#         result = particle(population1, state)
#         assert isinstance(result, kaku.Individual)

#     def test_global_particle_smooth_outputs_individual_after_two(self):
#         torch.manual_seed(1)

#         state = kaku.State()
#         particle = _particle.GlobalParticleSmooth(0)
#         population1 = kaku.Population(x=torch.rand(4, 4, 2))
#         population1.report(Assessment(torch.tensor([0.1, 0.2, 0.8, 1.0])))
#         result = particle(population1, state)
#         population2 = kaku.Population(x=torch.rand(4, 4, 2))
#         population2.report(Assessment(torch.tensor([0.1, 0.2, 0.8, 1.0])))
#         result2 = particle(population1, state)
#         assert isinstance(result2, kaku.Individual)

