import torch
from zenkai.memory._memory import BatchMemory


class TestBatchMemory(object):


    def test_add_to_memory_increases_size(self):
        memory = BatchMemory(samples=['k'])

        memory.add_batch(k=torch.randn(2, 2))
        assert len(memory) == 2

    def test_index_memory_return_correct_size(self):

        memory = BatchMemory(samples=['k'], singular=['w'])
        memory.add_batch(
            k=torch.randn(2, 2), w=torch.randn(2, 2)
        )
        memory.add_batch(
            k=torch.randn(3, 2), w=torch.randn(2, 2)
        )
        data = memory[1]
        assert data['k'].shape == torch.Size([2])
        assert data['w'].shape == torch.Size([2, 2])

    def test_index_memory_return_correct_size_when_indexing_multiple(self):

        memory = BatchMemory(samples=['k'], singular=['w'])
        memory.add_batch(
            k=torch.randn(2, 2), w=torch.randn(2, 2)
        )
        memory.add_batch(
            k=torch.randn(3, 2), w=torch.randn(2, 2)
        )
        data = memory[[1, 2]]
        assert data['k'].shape == torch.Size([2, 2])
        assert data['w'].shape == torch.Size([2, 2, 2])

    def test_n_batches_returns_correct_number_of_batches(self):

        memory = BatchMemory(samples=['k'], singular=['w'])
        memory.add_batch(
            k=torch.randn(2, 2), w=torch.randn(2, 2)
        )
        memory.add_batch(
            k=torch.randn(3, 2), w=torch.randn(2, 2)
        )
        assert memory.n_batches == 2

    def test_remove_from_memory_removes_samples(self):

        memory = BatchMemory(samples=['k'], singular=['w'])
        memory.add_batch(
            k=torch.randn(2, 2), w=torch.randn(2, 2)
        )
        memory.add_batch(
            k=torch.randn(3, 2), w=torch.randn(2, 2)
        )
        memory.remove_samples([1])

        assert len(memory) == 4

    def test_remove_from_memory_removes_batches_if_all_gone(self):

        memory = BatchMemory(samples=['k'], singular=['w'])
        memory.add_batch(
            k=torch.randn(2, 2), w=torch.randn(2, 2)
        )
        memory.add_batch(
            k=torch.randn(3, 2), w=torch.randn(2, 2)
        )
        memory.remove_samples([0, 1])

        assert memory.n_batches == 1

    def test_remove_random_samples_removes(self):

        memory = BatchMemory(samples=['k'], singular=['w'])
        memory.add_batch(
            k=torch.randn(2, 2), w=torch.randn(2, 2)
        )
        memory.add_batch(
            k=torch.randn(3, 2), w=torch.randn(2, 2)
        )
        memory.remove_random_samples(2)

        assert memory.n_samples == 3

    def test_remove_batch_removes_batch(self):

        memory = BatchMemory(samples=['k'], singular=['w'])
        memory.add_batch(
            k=torch.randn(2, 2), w=torch.randn(2, 2)
        )
        memory.add_batch(
            k=torch.randn(3, 2), w=torch.randn(2, 2)
        )
        memory.remove_batch(0)

        assert memory.n_samples == 3
        assert memory.n_batches == 1
