from gymnasium.spaces import Box
from torch import Tensor, as_tensor, autograd


class MinMaxStraightThroughFunction(autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, low: Tensor, high: Tensor) -> Tensor:
        return (x - low) / (high - low)

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple[Tensor, None, None]:
        return grad_output, None, None


def min_max_scaling(x: Tensor, space: Box, straight_through: bool = False) -> Tensor:
    """Normalize a tensor x to the range `[0, 1]` given a `Box` space.

    Args:
        x: The tensor to normalize.
        space: The `Box` space to normalize the tensor from.
        straight_through: If `True`, the gradient will be passed through the normalization (instead
            of backpropagating it through the normalization calculations).

    Returns:
        The normalized tensor.
    """
    # check if low and high are correctly set
    low_np = space.low
    high_np = space.high
    if (low_np >= high_np).any():
        raise ValueError("The low bound must be less than the high bound.")
    if not space.is_bounded():
        raise ValueError("The low and high bounds must not be infinite.")

    low = as_tensor(low_np, dtype=x.dtype, device=x.device)
    high = as_tensor(high_np, dtype=x.dtype, device=x.device)
    if straight_through:
        return MinMaxStraightThroughFunction.apply(x, low, high)  # type: ignore
    return (x - low) / (high - low)
