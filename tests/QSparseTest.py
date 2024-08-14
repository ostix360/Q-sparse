import unittest

from q_sparse.QSparse import *


class TestSTEFunction(unittest.TestCase):
    def test_STEFunction_forward_pass(self):
        input_tensor = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        mask_tensor = torch.tensor([1.0, 0.0, 1.0])
        result = STEFunction.apply(input_tensor, mask_tensor)
        expected = torch.tensor([1.0, 0.0, 3.0])
        self.assertTrue(torch.equal(result, expected))

    def test_STEFunction_backward_pass(self):
        input_tensor = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        mask_tensor = torch.tensor([1.0, 0.0, 1.0])
        result = STEFunction.apply(input_tensor, mask_tensor)
        result.sum().backward()
        expected_grad = torch.tensor([1.0, 1.0, 1.0])
        self.assertTrue(torch.equal(input_tensor.grad, expected_grad))

class TestTopKSparsity(unittest.TestCase):
    def test_TopKSparsity_forward_pass(self):
        module = TopKSparsity(k_ratio=0.5)
        input_tensor = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        result = module(input_tensor)
        expected = torch.tensor([[0.0, 0.0, 0.6, 0.8]])
        self.assertTrue(torch.allclose(result, expected, atol=1e-2))

    def test_TopKSparsity_batch_forward_pass(self):
        module = TopKSparsity(k_ratio=0.5)
        input_tensor = torch.tensor([[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]])
        result = module(input_tensor)
        expected = torch.tensor([[0.0, 0.0, 0.6, 0.8], [0.0, 0.0, 0.6, 0.8]])
        self.assertTrue(torch.allclose(result, expected, atol=1e-2))

    def test_TopKSparsity_edge_case_empty_tensor(self):
        module = TopKSparsity(k_ratio=0.5)
        input_tensor = torch.tensor([])
        result = module(input_tensor)
        expected = torch.tensor([])
        self.assertTrue(torch.equal(result, expected))

class TestQSparseLinear(unittest.TestCase):
    def test_QSparseLinear_forward_pass(self):
        module = QSparseLinear(in_features=4, out_features=2, k_ratio=0.5)
        input_tensor = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        result = module(input_tensor)
        self.assertEqual(result.shape, (1, 2))

    def test_QSparseLinear_edge_case_zero_input(self):
        module = QSparseLinear(in_features=4, out_features=2, k_ratio=0.5, bias=False)
        input_tensor = torch.zeros((1, 4))
        result = module(input_tensor)
        expected = torch.zeros((1, 2))
        self.assertTrue(torch.equal(result, expected))

class TestSquaredReLU(unittest.TestCase):
    def test_SquaredReLU_forward_pass(self):
        module = SquaredReLU()
        input_tensor = torch.tensor([-1.0, 0.0, 1.0])
        result = module(input_tensor)
        expected = torch.tensor([0.0, 0.0, 1.0])
        self.assertTrue(torch.equal(result, expected))

    def test_SquaredReLU_edge_case_large_values(self):
        module = SquaredReLU()
        input_tensor = torch.tensor([1e6, -1e6])
        result = module(input_tensor)
        expected = torch.tensor([1e12, 0.0])
        self.assertTrue(torch.equal(result, expected))

if __name__ == '__main__':
    unittest.main()