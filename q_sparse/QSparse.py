import torch
from torch import nn


class STEFunction(torch.autograd.Function):
    """
    A custom autograd function for the Straight-Through Estimator (STE).
    This function allows gradients to pass through unchanged during the backward pass.
    """

    @staticmethod
    def forward(ctx, input, mask):
        """
        Forward pass for the STE function.

        Args:
            ctx: Context object to store information for backward computation.
            input (torch.Tensor): Input tensor.
            mask (torch.Tensor): Mask tensor to apply element-wise multiplication.

        Returns:
            torch.Tensor: Result of element-wise multiplication of input and mask.
        """
        return input * mask

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for the STE function.

        Args:
            ctx: Context object with information from the forward pass.
            grad_output (torch.Tensor): Gradient of the loss with respect to the output.

        Returns:
            tuple: Gradient of the loss with respect to the input and None for the mask.
        """
        return grad_output, None  # STE: pass gradient through without modification as mentioned in the paper


class TopKSparsity(nn.Module):
    """
    A module that applies top-k sparsity to the input tensor.
    """

    def __init__(self, k_ratio):
        """
        Initializes the TopKSparsity module.

        Args:
            k_ratio (float): Ratio of elements to keep (top-k).
        """
        super().__init__()
        self.k_ratio = k_ratio

    def forward(self, x):
        """
        Forward pass for the TopKSparsity module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Sparsified and normalized tensor.
        """
        k = int(self.k_ratio * x.shape[-1])
        _, top_k = torch.topk(torch.abs(x), k, dim=-1)
        # Create a mask for the top-k values
        mask = torch.zeros_like(x)
        mask[...,top_k] = 1
        x = STEFunction.apply(x, mask)
        x = x / (torch.norm(x, p=2, dim=-1, keepdim=True) + 1e-6)
        return x


class QSparseLinear(nn.Linear):
    """
    A linear layer with integrated top-k sparsity.
    """

    def __init__(self, in_features, out_features, k_ratio=0.1, *args, **kwargs):
        """
        Initializes the QSparseLinear layer.

        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
            bias (bool): If set to False, the layer will not learn an additive bias (default: True).
            k_ratio (float): Ratio of elements to keep (top-k).
        """
        super().__init__(in_features, out_features, *args, **kwargs)
        self.k_ratio = k_ratio
        self.sparsity = TopKSparsity(k_ratio)

    def forward(self, x):
        """
        Forward pass for the QSparseLinear layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying sparsity and linear transformation.
        """
        x = self.sparsity(x)
        return super().forward(x)


class SquaredReLU(nn.Module):
    """
    A module that applies the ReLU activation function and then squares the result.
    """

    def __init__(self):
        """
        Initializes the SquaredReLU module.
        """
        super().__init__()
        self.act = nn.ReLU()

    def forward(self, x):
        """
        Forward pass for the SquaredReLU module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying ReLU and squaring the result.
        """
        return self.act(x) ** 2
