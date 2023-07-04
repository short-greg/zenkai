import torch

from zenkai.sensei import reporting
from .test_base import SampleMaterial


class TestEntry:

    def test_entry_logs_one_input(self):
        entry = reporting.Entry(2, 0)
        entry.add({'x': torch.tensor([2,3])})
        assert len(entry.data) == 1

    def test_entry_logs_one_input_and_converts_to_df(self):
        entry = reporting.Entry(2, 0)
        entry.add({'x': torch.tensor([2,3])})
        assert len(entry.df) == 1

    def test_entry_df_returns_correct_tensor(self):
        entry = reporting.Entry(2, 0)
        x = torch.tensor([2,3])
        entry.add({'x': x})
        assert (entry.df.loc[0, 'x'] == x).all()

    def test_entry_df_returns_correct_tensor_after_two_inputs(self):
        entry = reporting.Entry(2, 0)
        x = torch.tensor([2,3])
        x2 = torch.tensor([3,4])
        entry.add({'x': x})
        entry.add({'x': x2})
        assert (entry.df.loc[1, 'x'] == x2).all()

    def test_entry_df_returns_correct_tensor_after_add_two_inputs(self):
        entry = reporting.Entry(2, 0)
        x = torch.tensor([2,3])
        x2 = torch.tensor([3,4])
        entry.add({'x': x, 'x2': x2})
        assert (entry.df.loc[0, 'x2'] == x2).all()

    def test_entry_df_returns_correct_len(self):
        entry = reporting.Entry(2, 0)
        x = torch.tensor([2,3])
        x2 = torch.tensor([3,4])
        entry.add({'x': x})
        entry.add({'x': x2})
        assert len(entry) == 2


class TestLog:
    
    def test_get_entry_after_adding_one(self):

        log = reporting.Log("Trainer")
        id = log.add_entry(4)
        log.update_entry(id, {'x': torch.tensor([2,3])})
        assert len(log.entries[id]) == 1
    
    def test_get_df_after_adding_one(self):

        log = reporting.Log("Trainer")
        id = log.add_entry(4)
        log.update_entry(id, {'x': torch.tensor([2,3])})
        assert len(log.df) == 1
    
    def test_get_df_after_updating_entry_twice(self):

        log = reporting.Log("Trainer")
        id = log.add_entry(4)
        log.update_entry(id, {'x': torch.tensor([2,3])})
        log.update_entry(id, {'x': torch.tensor([4,3])})

        assert len(log.df) == 2

    def test_get_df_after_adding_two_entries(self):

        log = reporting.Log("Trainer")
        id = log.add_entry(4)
        log.update_entry(id, {'x': torch.tensor([2,3])})
        log.update_entry(id, {'x': torch.tensor([1,3])})
        id = log.add_entry(4)
        log.update_entry(id, {'x': torch.tensor([4,3])})

        assert len(log.df) == 3

    def test_current_returns_current_entry(self):

        log = reporting.Log("Trainer")
        log.add_entry(4)
        id = log.add_entry(4)
        assert log.current is log.entries[id]


class TestRecord:
    
    def test_add_entry_creates_new_entry(self):

        record = reporting.Record()
        id = record.add_entry("Trainer", 4)
        record.update_entry("Trainer", id, {'x': torch.tensor([2,3])})
        assert len(record.current("Trainer")) == 1

    def test_update_entry_twice_creates_two_entries(self):

        record = reporting.Record()
        id = record.add_entry("Trainer", 4)
        record.update_entry("Trainer", id, {'x': torch.tensor([2,3])})
        record.update_entry("Trainer", id, {'x': torch.tensor([4,3])})
        assert len(record.current("Trainer")) == 2
    
    def test_update_entry_twice_creates_two_entries(self):

        record = reporting.Record()
        id = record.add_entry("Trainer", 4)
        record.update_entry("Trainer", id, {'x': torch.tensor([2,3])})
        record.update_entry("Trainer", id, {'x': torch.tensor([4,3])})
        id2 = record.add_entry("TrainerX", 4)
        record.update_entry("TrainerX", id2, {'x': torch.tensor([1,3])})
        assert len(record.df()) == 3


class TestLogger:
    
    def test_add_entry_creates_new_entry(self):

        record = reporting.Record()
        logger = record.create_logger("Trainer", SampleMaterial())
        logger({'x': torch.tensor([2,3])})
        logger({'x': torch.tensor([4,3])})
        assert len(logger.log) == 1

    def test_add_entry_creates_new_entry(self):

        record = reporting.Record()
        logger = record.create_logger("Trainer", SampleMaterial())
        logger({'x': torch.tensor([2,3])})
        logger.advance()
        logger({'x': torch.tensor([4,3])})
        assert len(logger.log) == 2

    # def test_update_entry_twice_creates_two_entries(self):

    #     record = reporting.Record()
    #     id = record.add_entry("Trainer", 4)
    #     record.update_entry("Trainer", id, {'x': torch.tensor([2,3])})
    #     record.update_entry("Trainer", id, {'x': torch.tensor([4,3])})
    #     assert len(record.current("Trainer")) == 2
    
    # def test_update_entry_twice_creates_two_entries(self):

    #     record = reporting.Record()
    #     id = record.add_entry("Trainer", 4)
    #     record.update_entry("Trainer", id, {'x': torch.tensor([2,3])})
    #     record.update_entry("Trainer", id, {'x': torch.tensor([4,3])})
    #     id2 = record.add_entry("TrainerX", 4)
    #     record.update_entry("TrainerX", id2, {'x': torch.tensor([1,3])})
    #     assert len(record.df()) == 3



class TestResults:

    def test_add_twice_results_in_two_results(self):
        results = reporting.Results()
        results.add(
            {"loss" :torch.tensor(0.2)}
        )
        assert len(results['loss']) == 1

    def test_add_result(self):
        results = reporting.Results()
        results.add(
            {"loss" :torch.tensor(0.2)}
        )
        results.add(
            {"loss" :torch.tensor(0.3)}
        )
        assert len(results['loss']) == 2

    def test_aggregate_aggregates_results(self):
        results = reporting.Results(2)
        results.add(
            {"loss" :torch.tensor(0.2)}
        )
        results.add(
            {"loss" :torch.tensor(0.3)}
        )
        results.add(
            {"loss" :torch.tensor(0.4)}
        )
        assert round(results.aggregate('loss')['loss'], 2) == 0.35

