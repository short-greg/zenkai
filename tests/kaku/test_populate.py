import pytest
import torch

from zenkai.kaku import Individual, Population
from zenkai.utils import get_model_parameters

from zenkai import Assessment


@pytest.fixture
def x1():
    generator = torch.Generator()
    generator.manual_seed(1)
    return torch.rand(2, 2, generator=generator)


@pytest.fixture
def binary_x():
    generator = torch.Generator()
    generator.manual_seed(1)
    return torch.rand(2, 2, generator=generator).sign()


@pytest.fixture
def binary_x2():
    generator = torch.Generator()
    generator.manual_seed(1)
    return (torch.rand(2, 2, generator=generator) > 0.5).float()


@pytest.fixture
def pop_x1():
    generator = torch.Generator()
    generator.manual_seed(1)
    return torch.rand(3, 2, 2, generator=generator)


@pytest.fixture
def pop_x2():
    generator = torch.Generator()
    generator.manual_seed(3)
    return torch.rand(3, 4, 2, generator=generator)


@pytest.fixture
def x2():
    generator = torch.Generator()
    generator.manual_seed(2)
    return torch.rand(2, 2, generator=generator)


@pytest.fixture
def model1():
    return torch.nn.Linear(2, 3)


@pytest.fixture
def model2() -> torch.nn.Module:
    return torch.nn.Linear(2, 3)


@pytest.fixture
def p1() -> torch.nn.parameter.Parameter:
    generator = torch.Generator()
    generator.manual_seed(3)
    return torch.nn.parameter.Parameter(torch.rand(2, 2, generator=generator))


@pytest.fixture
def individual1(x1) -> Individual:
    return Individual(x=x1)


@pytest.fixture
def binary_individual1(binary_x) -> Individual:
    return Individual(x=binary_x)


@pytest.fixture
def binary_individual2(binary_x2) -> Individual:
    return Individual(x=binary_x2)


@pytest.fixture
def individual_model(model1) -> Individual:
    return Individual(model=model1)


@pytest.fixture
def assessment1() -> Assessment:
    return Assessment(torch.rand(2, 3))


@pytest.fixture
def population1(pop_x1) -> Population:
    return Population(x=pop_x1)


@pytest.fixture
def population1_with_assessment(pop_x1) -> Population:
    return Population(x=pop_x1).report(Assessment(torch.rand(pop_x1.size(0)).cumsum(0)))


@pytest.fixture
def population2_with_assessment(pop_x2) -> Population:
    return Population(x=pop_x2).report(
        Assessment(torch.rand(pop_x2.shape[:2]).cumsum(0))
    )


@pytest.fixture
def binary_population2_with_assessment(pop_x2) -> Population:
    return Population(x=pop_x2.sign()).report(
        Assessment(torch.rand(pop_x2.shape[:2]).cumsum(0))
    )


@pytest.fixture
def individual2(x1, x2) -> Individual:
    return Individual(x1=x1, x2=x2)


class TestIndividual: 
    def test_get_returns_requested_parameter(self, x1, individual1):
        assert (x1 == individual1["x"]).all()

    def test_set_parameter_updates_parameter(self, x1, individual1, p1):
        individual1.set_p(p1, "x")
        assert (p1.data == x1).all()

    def test_using_model_in_individual_retrieves_parameters(
        self, individual_model: Individual, model1, model2
    ):
        individual_model.set_model(model2, "model")
        assert (get_model_parameters(model2) == get_model_parameters(model1)).all()

    def test_iter_returns_all_individuals(self, x1, x2, individual2):
        result = {k: v for k, v in individual2.items()}
        assert result["x1"] is x1
        assert result["x2"] is x2

    def test_report_sets_the_assessment(self, individual1, assessment1):
        individual1 = individual1.report(assessment1)
        assert individual1.assessment is assessment1

    def test_add_individuals_adds_correctly(self, x1, x2, individual2):
        individual3 = individual2 + individual2
        assert (individual3["x1"] == (x1 + x1)).all()
        assert (individual3["x2"] == (x2 + x2)).all()

    def test_add_individuals_adds_correctly_when_not_exists(
        self, x1, x2, individual1, individual2
    ):
        individual1 = individual1.copy()
        individual1["x1"] = individual1["x"]
        individual3 = individual1 + individual2

        assert (individual3["x1"] == (x1 + x1)).all()
        assert (individual3["x2"] == x2).all()

    def test_sub_individuals_subs_correctly_when_not_exists(
        self, x1, x2, individual1, individual2
    ):
        individual1 = individual1.copy()
        individual1["x1"] = x2
        
        individual3 = individual1 - individual2

        assert (individual3["x1"] == (x2 - x1)).all()
        assert individual3.get("x2") is None

    def test_mul_individuals_muls_correctly_when_not_exists(
        self, x1, x2, individual1, individual2
    ):
        individual1 = individual1.copy()
        individual1["x1"] = x2
        individual3 = individual1 * individual2

        assert (individual3["x1"] == (x2 * x1)).all()
        assert individual3.get("x2") is None

    def test_le_gets_less_value_from_inviduals(self, individual1):
        individual1 = individual1.copy()
        individual3 = individual1 < individual1 + 1

        assert (individual3["x"]).all()

    def test_lt_gets_less_value_from_inviduals(self, individual1):
        individual1 = individual1.copy()
        individual2 = individual1.copy()
        individual3 = individual1 <= individual2

        assert (individual3["x"]).all()

    def test_gte_gets_greater_value_from_inviduals(self, individual1):
        individual1 = individual1.copy()
        individual2 = individual1.copy()
        individual3 = individual1 >= individual2

        assert (individual3["x"]).all()

    def test_gt_gets_greater_value_from_inviduals(self, individual1):
        individual2 = individual1 - 1
        individual3 = individual1 > individual2
        assert (individual3["x"]).all()

    def test_eq_returns_equal_if_same(self, individual1):
        individual1 = individual1.copy()
        individual2 = individual1.copy()

        assert (individual1 == individual2)["x"].all()

    def test_loop_over_loops_over_both_individuals(
        self, individual1: Individual, individual2: Individual
    ):

        keys = ("x", "x1", "x2")
        for key, mine, other in individual1.loop_over(individual2):
            if key == "x":
                assert other is None
            if key == "x1" or key == "x2":
                assert mine is None
            assert key in keys

    def test_loop_over_loops_over_both_individuals_intersection(
        self, individual1: Individual, individual2: Individual
    ):

        keys = "x"
        individual2["x"] = individual1["x"]
        for key, mine, other in individual1.loop_over([individual2], False, False):
            if key == "x":
                assert other is not None and mine is not None
            assert key in keys

    def test_loop_over_loops_over_only_mine(
        self, individual1: Individual, individual2: Individual
    ):

        keys = "x"
        for key, mine, other in individual1.loop_over([individual2], True, True):
            if key == "x":
                assert other is None and mine is not None
            assert key in keys

    def test_apply_creates_rand_tensors_like(
        self, individual1: Individual, individual2: Individual
    ):

        individual2 = individual1.apply(torch.rand_like, ["x"])
        assert "x" in individual2

    def test_apply_filters_out_y(
        self, individual1: Individual, individual2: Individual
    ):

        individual1["y"] = torch.rand(4)
        individual2 = individual1.apply(torch.rand_like, ["x"], filter_keys=True)
        assert "y" not in individual2

    def test_apply_does_not_filter_out_y(
        self, individual1: Individual, individual2: Individual
    ):

        individual1["y"] = torch.rand(4)
        individual2 = individual1.apply(torch.rand_like, ["x"], filter_keys=True)
        assert "y" not in individual2

    def test_binary_op_adds_two(self, individual1: Individual, individual2: Individual):

        individual1["x"] = torch.rand(4)
        individual2 = individual1.apply(torch.rand_like, ["x"], filter_keys=True)
        assert "y" not in individual2

    def test_validate_keys_returns_false_if_different(
        self, individual1: Individual, individual2: Individual
    ):

        assert not individual1.validate_keys(individual2)

    def test_validate_keys_returns_true_if_same(self, individual1: Individual):

        assert individual1.validate_keys(individual1)

    def test_spawn_returns_item_of_same_type(
        self, individual1: Individual, individual2: Individual
    ):

        individual3 = individual1.spawn(individual2)
        assert individual3["x1"] is individual2["x1"]

    def test_copy_copys_the_individual(self, individual1: Individual):

        individual3 = individual1.copy()
        assert individual3["x"] is individual1["x"]

    def test_clone_creates_copy_of_the_data(self, individual1: Individual):

        individual3 = individual1.clone()
        assert not individual3["x"] is individual1["x"]

    def test_clone_creates_copy_of_the_data_that_is_equal(
        self, individual1: Individual
    ):

        individual3 = individual1.clone()
        assert (individual3["x"] == individual1["x"]).all()


class TestPopulation:
    def test_get_returns_population_parameter(self, pop_x1, population1):
        assert population1["x"] is pop_x1

    def test_sets_returns_population_parameter_for_x(self, pop_x1, x1, population1):
        x1 = torch.clone(x1)
        assert (pop_x1[0] == population1["x", 0]).all()

    def test_cannot_set_if_sizes_are_not_the_same(self):
        with pytest.raises(ValueError):
            Population(x=torch.rand(3, 2, 2), y=torch.rand(2, 3, 3))

    def test_len_is_correct(self):
        population = Population(x=torch.rand(3, 2, 2), y=torch.rand(3, 3, 3))
        assert population.k == 3

    def test_loop_over_population_returns_three_individuals(self):
        population = Population(x=torch.rand(3, 2, 2), y=torch.rand(3, 3, 3))
        individuals = list(population.individuals())
        assert len(individuals) == 3
        assert isinstance(individuals[0], Individual)

    def test_set_model_sets_the_model_correctly(self):
        linear = torch.nn.Linear(3, 3)
        population = Population(x=torch.rand(3, 2, 2), y=torch.rand(3, 12))
        population.set_model(linear, "y", 0)
        assert (get_model_parameters(linear) == population["y", 0]).all()

    def test_set_p_sets_parameters_correctly(self, p1, pop_x1):
        population = Population(x=pop_x1)
        population.set_p(p1, "x", 1)
        assert (p1 == pop_x1[1]).all()

    def test_gather_sub_gets_correct_output_when_dim_is_3(self):

        x1 = torch.rand(4, 2, 2)
        gather_by = torch.randint(0, 4, (6, 2))

        population = Population(x=x1)
        gathered = population.gather_sub(gather_by)
        assert gathered["x"].size() == torch.Size([6, 2, 2])

    def test_gather_sub_gets_correct_output_when_dim_is_2(self):

        x1 = torch.rand(4, 2)
        gather_by = torch.randint(0, 4, (6, 2))

        population = Population(x=x1)
        gathered = population.gather_sub(gather_by)
        assert gathered["x"].size() == torch.Size([6, 2])

    def test_gather_sub_gets_correct_output_when_dim_is_4(self):

        x1 = torch.rand(4, 2, 2, 3)
        gather_by = torch.randint(0, 4, (6, 2))

        population = Population(x=x1)
        gathered = population.gather_sub(gather_by)
        assert gathered["x"].size() == torch.Size([6, 2, 2, 3])

    def test_gather_sub_raises_error_if_dim_too_large(self):

        x1 = torch.rand(4, 2)
        gather_by = torch.randint(0, 4, (6, 2, 2))

        population = Population(x=x1)
        with pytest.raises(ValueError):
            population.gather_sub(gather_by)

    def test_lt_gets_greater_value_from_inviduals(self):
        population = Population(x=torch.rand(3, 2, 2))
        population2 = Population(x=population["x"] + 0.1)
        assert (population < population2)["x"].all()

    def test_apply_rand_like_produces_similar_population(self):
        population = Population(x=torch.rand(3, 2, 2))
        population2 = population.apply(torch.rand_like)

        assert (population2["x"] != population["x"]).any()

    def test_apply_add_produces_population_plus_1(self):
        population = Population(x=torch.rand(3, 2, 2))
        population2 = population.apply(lambda x: torch.add(x, 1))

        assert (population2["x"] == population["x"] + 1).all()

    def test_apply_add_does_not_add_if_filtered(self):
        population = Population(x=torch.rand(3, 2, 2))
        population2 = population.apply(lambda x: torch.add(x, 1), keys=["y"])

        assert (population2["x"] == population["x"]).all()

    def test_apply_add_does_not_add_if_not_filtered(self):
        population = Population(x=torch.rand(3, 2, 2))
        population2 = population.apply(lambda x: torch.add(x, 1), keys=["x"])

        assert (population2["x"] == population["x"] + 1).all()

    def test_k_returns_size_of_population(self):
        population = Population(x=torch.rand(3, 2, 2))
        assert population.k == 3

    def test_individuals_returns_three_individuals(self):
        population = Population(x=torch.rand(3, 2, 2))
        assert len(list(population.individuals())) == 3

    def test_individuals_returns_correct_individual(self):
        population = Population(x=torch.rand(3, 2, 2))
        individuals = list(population.individuals())
        assert (individuals[1]["x"] == population["x"][1]).all()
        assert (individuals[2]["x"] == population["x"][2]).all()

    def test_get_i_returns_correct_individual(self):
        population = Population(x=torch.rand(3, 2, 2))
        individual = population.get_i(2)
        assert (individual["x"] == population["x"][2]).all()

    def test_select_gets_correct(self):
        population = Population(
            x=torch.rand(3, 2, 2), y=torch.rand(3, 3, 2), z=torch.rand(3, 3, 2)
        )
        sub_population = population.select(["x", "y"])
        assert "z" not in sub_population
        assert population["x"] is sub_population["x"]

    def test_select_raises_error_if_name_not_in_population(self):
        population = Population(
            x=torch.rand(3, 2, 2), y=torch.rand(3, 3, 2), z=torch.rand(3, 3, 2)
        )
        with pytest.raises(KeyError):
            population.select(["t"])

    def test_sub_gets_a_sub_population(self):
        population = Population(x=torch.rand(3, 2, 2))
        sub = population.sub[[1, 0]]
        assert (sub["x"][0] == population["x"][1]).all()
        assert (sub["x"][1] == population["x"][0]).all()
