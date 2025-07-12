"""
This module contains the fixtures for the tests.
"""

# 3pps
import torch
import pytest

# Own modules
from src.utils import set_seed
from tests.utils import add_seed


@pytest.fixture(params=[*add_seed((64, 10, 5, 1)), *add_seed((32, 20, 15, 10))])
def artifacts(request) -> tuple[torch.Tensor, torch.Tensor, torch.nn.Module]:
    """
    This function is a fixture to define example random inputs.

    Args:
        request: Argument containing the introduced arguments.

    Returns:
        Inputs tensor. Dimensions: [batch, channels, height, width].
    """

    # Get parameters
    batch_size: int
    input_dim: int
    hidden_dim: int
    output_dim: int
    seed: int
    batch_size, input_dim, hidden_dim, output_dim, seed = request.param

    # Set seed
    set_seed(seed)

    # define model
    model: torch.nn.Module = torch.nn.Sequential(
        torch.nn.Linear(input_dim, hidden_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_dim, output_dim),
    ).double()

    # Init parameters
    for parameter in model.parameters():
        parameter.data *= 100

    # Define inputs and targets
    inputs: torch.Tensor = torch.rand(batch_size, input_dim).double()
    targets: torch.Tensor = (
        torch.rand(batch_size, output_dim).double()
    )
    
    # Get inputs
    inputs *= 100
    targets *= 100

    return inputs, targets, model
