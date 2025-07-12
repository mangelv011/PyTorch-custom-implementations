"""
This module contains the code to test the models module.
"""

# 3pps
import torch
import pytest

# own modules
from src.utils import set_seed
from src.models import Dropout


@pytest.mark.order(6)
@pytest.mark.parametrize("p, seed", [(0.0, 0), (0.5, 1), (0.7, 2)])
def test_dropout(
    p: float, seed: int, artifacts: tuple[torch.Tensor, torch.Tensor, torch.nn.Module]
) -> None:
    """
    This function tests the Dropout layer.

    Args:
        p: dropout probability.
        seed: seed for test.

    Returns:
        None.
    """

    # define inputs
    inputs: torch.Tensor = artifacts[0]
    inputs_torch: torch.Tensor = inputs.clone()

    # define dropout
    dropout = Dropout(p)
    dropout_torch: torch.nn.Module = torch.nn.Dropout(p)

    # activate train mode
    dropout.train()
    dropout_torch.train()

    # compute outputs
    set_seed(seed)
    outputs: torch.Tensor = dropout(inputs)
    set_seed(seed)
    outputs_torch: torch.Tensor = dropout_torch(inputs)

    # check output type
    assert isinstance(
        outputs, torch.Tensor
    ), f"Incorrect type, expected torch.Tensor got {type(outputs)}"

    # check output size
    assert (
        outputs.shape == inputs.shape
    ), f"Incorrect shape, expected {inputs.shape}, got {outputs.shape}"

    # check outputs of dropout
    assert torch.allclose(outputs, outputs_torch), (
        "Incorrect outputs when train mode activated, outputs are not equal to "
        "pytorch implementation"
    )

    # activate eval mode
    dropout.eval()
    dropout_torch.eval()

    # Compute outputs
    set_seed(seed)
    outputs = dropout(inputs)
    set_seed(seed)
    outputs_torch = dropout_torch(inputs)

    # check outputs of dropout
    assert torch.allclose(outputs, outputs_torch), (
        "Incorrect outputs when eval mode activated, outputs are not equal to "
        "pytorch implementation"
    )

    # define dropout with inplace
    dropout = Dropout(p, inplace=True)
    dropout_torch = torch.nn.Dropout(p, inplace=True)

    # compute outputs
    set_seed(seed)
    dropout(inputs)
    set_seed(seed)
    dropout_torch(inputs_torch)

    # check outputs of dropout
    assert torch.allclose(inputs, inputs_torch), (
        "Incorrect outputs when inplace is activated, outputs are not equal to "
        "pytorch implementation"
    )

    return None
