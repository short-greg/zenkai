

import torch


# 1st party
import typing
from abc import ABC, abstractmethod
import functools

# 3rd party
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from ..kaku import IndexMap, Selector

# 1st party
from abc import ABC, abstractmethod
import typing

# 3rd party
import torch

# local
from ..kaku import State, Population, Individual, TensorDict
from ..utils import gahter_idx_from_population


# local
from ..utils import get_model_parameters, update_model_parameters, expand_dim0, flatten_dim0, gather_idx_from_population

from ..kaku import IO, Assessment
from ..kaku import Reduction, Criterion, State, Criterion

from copy import deepcopy


import torch

from ..kaku.assess import Assessment
from abc import abstractmethod, ABC

# TODO: Move to utils


# Only use a class if I think that it will be 'replaceable'
# Elitism() <-
# Mixer() <- remove   tansaku.conserve(old_p, new_p, prob=...)
# Crossover()
# Perturber()
# Sampler() (Include reduers in here)
# SlopeCalculator() <- doesn't need to be a functor.. I should combine this with "SlopeMapper"... Think about this more
# concat <- add in concat
# Limiter??? - similar to "keep mixer" -> tansaku.limit_feature(population, limit=...)
# Divider() -> ParentSelector() <- rename
# Assessor
# concat()
# 

