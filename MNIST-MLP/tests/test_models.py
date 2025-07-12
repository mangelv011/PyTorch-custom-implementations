# deep learning libraries
import torch

# other libraries
import pytest

# own modules
from src.models import ReLU, Linear, MyModel

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


@pytest.mark.order(2)
def test_relu() -> None:
    """
    This is the test for the relu class

    Raises:
        RuntimeError: Error with gradient computation in test
        RuntimeError: Error with gradient computation in test
        RuntimeError: Error with gradient computation in test
    """

    # defie relu class
    relu: torch.nn.Module = ReLU()

    # check positive forward
    example_input: torch.Tensor = (
        torch.FloatTensor(3, 3).uniform_(0.1, 10).requires_grad_(True)
    )
    example_output: torch.Tensor = relu(example_input)
    assert (
        example_input != example_output
    ).sum() == 0, "Incorrect forward for positive numbers"

    # check positive backward
    example_output.sum().backward()
    if example_input.grad is None:
        raise RuntimeError("Error with gradient computation in test")
    assert (
        example_input.grad != 1
    ).sum() == 0, "Incorrect backward for positive numbers"

    # check negative transformation
    example_input = torch.FloatTensor(3, 3).uniform_(-10, -0.1).requires_grad_(True)
    example_output = relu(example_input)
    assert (example_output != 0).sum() == 0, "Incorrect forward for negative numbers"

    # check negative backward
    example_output.sum().backward()
    if example_input.grad is None:
        raise RuntimeError("Error with gradient computation in test")
    assert (
        example_input.grad != 0
    ).sum() == 0, "Incorrect backward for negative numbers"

    # check backward on zero
    example_input = torch.zeros(3, 3).requires_grad_(True)
    example_output = relu(example_input)
    example_output.sum().backward()
    if example_input.grad is None:
        raise RuntimeError("Error with gradient computation in test")
    assert (example_input.grad != 0).sum() == 0, "Incorrect backward for zero"

    return None


@pytest.mark.order(3)
@pytest.mark.parametrize("input_dim, output_dim", [(5, 10)])
def test_linear(input_dim: int, output_dim: int) -> None:
    """
    This is the test for the linear class.

    Args:
        input_dim: dimension of the input.
        output_dim: dimension of he output.
    """

    # define linear class
    linear: torch.nn.Module = Linear(input_dim, output_dim)

    parameters: list[torch.nn.Parameter] = list(linear.parameters())

    # check number of parameters
    assert len(parameters) == 2, "Incorrect number of parameters, they must be 2"

    # define which parameter is the weights and which one is the bias
    if parameters[1].shape[0] == 1:
        weight_index: int = 0
        bias_index: int = 1

    else:
        weight_index = 1
        bias_index = 0

    # check shape of the weights
    assert parameters[weight_index].shape == (
        input_dim,
        output_dim,
    ), "Incorrect weight shape"

    # check shape of the bias
    assert parameters[bias_index].shape == (1, output_dim), "Incorrect bias shape"

    # get output
    output: torch.Tensor = linear(torch.rand(64, input_dim))

    # check object
    assert isinstance(output, torch.Tensor), "Incorrect output object type"

    # check shape of output
    assert output.shape == (64, output_dim), "Incorrect shape of output"

    return None


@pytest.mark.order(4)
@pytest.mark.parametrize(
    "input_size, output_size, hidden_sizes", [(728, 10, (128, 64, 32))]
)
def test_mymodel(
    input_size: int, output_size: int, hidden_sizes: tuple[int, ...]
) -> None:
    """
    This is the test for the MyModel class.

    Args:
        input_size: dimension of the input.
        output_size: dimension of the output.
        hidden_sizes: hidden dimensions.
    """

    # define mymodel class
    model: torch.nn.Module = MyModel(input_size, output_size, hidden_sizes).to(device)

    # get output
    output: torch.Tensor = model(torch.rand(64, input_size).to(device))

    # check object
    assert isinstance(output, torch.Tensor), "Incorrect output object type"

    # check shape of output
    assert output.shape == (64, output_size), "Incorrect shape of output"

    return None
