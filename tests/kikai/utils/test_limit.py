from zenkai.kikai.utils._limit import RandomFeatureIdxGen


class TestRandomChoiceLimitGen(object):

    def test_random_limiter_returns_four_elements(self):
        limiter = RandomFeatureIdxGen(10, 4)
        assert len(limiter()) == 4

    def test_random_limiter_returns_unique_elements(self):
        limiter = RandomFeatureIdxGen(10, 4)
        assert len(set(limiter().tolist())) == 4
