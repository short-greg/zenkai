# 1st party
from uuid import UUID, uuid1

# 3rd party
import torch.nn as nn
import torch as th
import torch.functional as F
import torch.nn.functional as FNN

# local
from zenkai.tako import core


class TestID:

    def test_id_is_uuid(self):
        id = core.ID()
        assert isinstance(id.x, UUID)
