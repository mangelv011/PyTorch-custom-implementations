# 3pps
import torch
import pytest

# own modules
from src.models import ReLU, Linear, Conv2d, Block, CNNModel
from src.utils import parameters_to_double, set_seed

# set seed and device
set_seed(42)


@pytest.mark.order(3)
def test_relu() -> None:
    """
    This function is the test for the relu function.
    """

    # define inputs
    inputs: torch.Tensor = torch.FloatTensor(3, 3).uniform_(-10, 10)
    inputs.requires_grad_(True)

    # define implemented relu
    model: torch.nn.Module = ReLU()

    # compute outputs and backward
    outputs = model(inputs)
    outputs.sum().backward()

    # get grads values
    if inputs.grad is None:
        assert False, "Gradients not returned, none value detected"
    inputs_grad: torch.Tensor = inputs.grad.clone()

    # define torch relu
    model_torch: torch.nn.Module = torch.nn.ReLU()

    # compute outputs and backward
    outputs_torch = model_torch(inputs)
    model_torch.zero_grad()
    inputs.grad.zero_()
    outputs_torch.sum().backward()

    # get grads values
    if inputs.grad is None:
        assert False, "Gradients not returned, none value detected"
    inputs_grad_torch: torch.Tensor = inputs.grad.clone()

    # check outputs
    assert (outputs != outputs_torch).sum().item() == 0, "Incorrect forward"

    # check inputs grads
    assert (
        inputs_grad != inputs_grad_torch
    ).sum().item() == 0, "Incorrect inputs gradients"

    # define inputs in zero
    inputs = torch.zeros(3, 3)
    inputs.requires_grad_(True)

    # define implemented relu
    model = ReLU()

    # compute outputs and backward
    outputs = model(inputs)
    outputs.sum().backward()

    # get grads values
    if inputs.grad is None:
        assert False, "Gradients not returned, none value detected"
    inputs_grad = inputs.grad.clone()

    # define torch relu
    model_torch = torch.nn.ReLU()

    # compute outputs and backward
    outputs_torch = model_torch(inputs)
    model_torch.zero_grad()
    inputs.grad.zero_()
    outputs_torch.sum().backward()

    # get grads values
    if inputs.grad is None:
        assert False, "Gradients not returned, none value detected"
    inputs_grad_torch = inputs.grad.clone()

    # check outputs
    assert (outputs != outputs_torch).sum().item() == 0, "Incorrect forward at 0"

    # check inputs grads
    assert (
        inputs_grad != inputs_grad_torch
    ).sum().item() == 0, "Incorrect inputs gradients at 0"

    return None


@pytest.mark.order(4)
def test_linear() -> None:
    """
    This function is the test for the linear model.
    """

    # define inputs
    inputs: torch.Tensor = torch.rand(64, 30).double()
    inputs.requires_grad_(True)

    # define linear
    set_seed(42)
    model: torch.nn.Module = Linear(30, 10)
    parameters_to_double(model)

    # compute outputs and backward
    outputs = model(inputs)
    outputs.sum().backward()

    # get grads values
    if model.weight.grad is None or model.bias.grad is None or inputs.grad is None:
        assert False, "Gradients not returned, none value detected"
    grad_weight: torch.Tensor = model.weight.grad.clone()
    grad_bias: torch.Tensor = model.bias.grad.clone()
    inputs_grad: torch.Tensor = inputs.grad.clone()

    # define torch linear
    set_seed(42)
    model_torch = torch.nn.Linear(30, 10)
    parameters_to_double(model_torch)

    # compute outputs and backward
    outputs_torch = model_torch(inputs)
    model.zero_grad()
    inputs.grad.zero_()
    outputs_torch.sum().backward()

    # get grads values
    if (
        model_torch.weight.grad is None
        or model_torch.bias.grad is None
        or inputs.grad is None
    ):
        assert False, "Gradients not returned, none value detected"
    grad_weight_torch: torch.Tensor = model_torch.weight.grad.clone()
    grad_bias_torch: torch.Tensor = model_torch.bias.grad.clone()
    inputs_grad_torch: torch.Tensor = inputs.grad.clone()

    # check foward
    assert (outputs != outputs_torch).sum().item() == 0, "Incorrect forward"

    # check weights grads
    assert (
        grad_weight != grad_weight_torch
    ).sum().item() == 0, "Incorrect weights gradients"

    # check bias grads
    assert (grad_bias != grad_bias_torch).sum().item() == 0, "Incorrect bias gradients"

    # check inputs grads
    assert (
        inputs_grad != inputs_grad_torch
    ).sum().item() == 0, "Incorrect inputs gradients"

    return None


@pytest.mark.order(5)
def test_conv() -> None:
    """
    This function is the test for the conv model.
    """

    # define inputs
    inputs: torch.Tensor = torch.rand(64, 3, 32, 32).double()
    inputs.requires_grad_(True)

    # define conv
    set_seed(42)
    model: torch.nn.Module = Conv2d(3, 10, 7)
    parameters_to_double(model)

    # compute outputs and backward
    outputs = model(inputs)
    outputs.sum().backward()

    # get grads
    if model.weight.grad is None or model.bias.grad is None or inputs.grad is None:
        assert False, "Gradients not returned, none value detected"
    grad_inputs: torch.Tensor = inputs.grad.clone()
    grad_weight: torch.Tensor = model.weight.grad.clone()
    grad_bias: torch.Tensor = model.bias.grad.clone()

    # define conv
    set_seed(42)
    model_torch: torch.nn.Module = torch.nn.Conv2d(3, 10, 7)
    parameters_to_double(model_torch)

    # compute outputs and backward
    outputs_torch = model_torch(inputs)
    inputs.grad.zero_()
    outputs_torch.sum().backward()

    # get grads
    if (
        model_torch.weight.grad is None
        or model_torch.bias.grad is None
        or inputs.grad is None
    ):
        assert False, "Gradients not returned, none value detected"
    grad_inputs_torch: torch.Tensor = inputs.grad.clone()
    grad_weight_torch: torch.Tensor = model_torch.weight.grad.clone()
    grad_bias_torch: torch.Tensor = model_torch.bias.grad.clone()

    # check values of the forward
    assert (outputs != outputs_torch).sum().item() == 0, "Incorrect forward"

    # check values of the inputs gradients
    assert (
        grad_inputs != grad_inputs_torch
    ).sum().item() == 0, "Incorrect inputs gradients"

    # check values of the weights gradients
    assert (
        grad_weight != grad_weight_torch
    ).sum().item() == 0, "Incorrect weights gradient"

    # check values of the bias gradients
    assert (grad_bias != grad_bias_torch).sum().item() == 0, "Incorrect bias gradients"

    return None


@pytest.mark.order(6)
@pytest.mark.parametrize(
    "input_channels, output_channels, stride", [(3, 10, 2), (1, 20, 4)]
)
def test_block(input_channels: int, output_channels: int, stride: int) -> None:
    """
    This is a test for the Block class.

    Args:
        input_channels: inut channels.
        output_channels: output channels.
        stride: stride.
    """

    # define block
    block: torch.nn.Module = Block(input_channels, output_channels, stride)

    # check sequential object
    assert isinstance(
        list(block.children())[0], torch.nn.Sequential
    ), "Layers not encapsulated inside sequential"

    # check sequential
    first_child: torch.nn.Module = list(block.children())[0]
    if not isinstance(first_child, torch.nn.Sequential):
        assert False, "Sequential not found"

    # define sequential
    sequential: torch.nn.Sequential = first_child

    # check length
    assert len(sequential) == 6, "Incorrect sequential length"

    # check sequential first element type
    assert isinstance(
        sequential[0], torch.nn.Conv2d
    ), f"Incorrect type of first element, expected conv got {type(sequential[0])}"

    # check stride 1 in first conv layer
    assert sequential[0].stride == (
        1,
        1,
    ), "Incorrect stride in first conv layer"

    # check sequential second element type
    assert isinstance(
        sequential[1], torch.nn.ReLU
    ), f"Incorrect type of second element, expected relu got {type(sequential[1])}"

    # check sequential third element type
    assert isinstance(
        sequential[2], torch.nn.Conv2d
    ), f"Incorrect type of third element, expected conv got {type(sequential[2])}"

    # check stride 2 in third conv layer
    assert sequential[2].stride == (
        stride,
        stride,
    ), "Incorrect stride in third conv layer"

    # check fourth element
    assert isinstance(
        sequential[3], torch.nn.ReLU
    ), f"Incorrect type of fourth element, expected relu got {type(sequential[3])}"

    # check sequential fifth element type
    assert isinstance(
        sequential[4], torch.nn.Conv2d
    ), f"Incorrect type of fifth element, expected conv got {type(sequential[4])}"

    # check stride 1 in fifth conv layer
    assert sequential[4].stride == (
        1,
        1,
    ), "Incorrect stride in fifth conv layer"

    # check sixth element
    assert isinstance(
        sequential[5], torch.nn.ReLU
    ), f"Incorrect type of sixth element, expected relu got {type(sequential[5])}"

    # define input and compute output
    example_input: torch.Tensor = torch.rand(64, input_channels, 224, 224)
    example_output: torch.Tensor = block(example_input)

    # check type of object
    assert isinstance(example_output, torch.Tensor), "Incorrect output object type"

    # check dimensions
    assert example_output.shape == (
        64,
        output_channels,
        224 // stride,
        224 // stride,
    ), "Incorrect shape of output object"

    return None


@pytest.mark.order(7)
def test_cnnmodel() -> None:
    """
    This is the function to test cnnmodel.
    """

    # compute output
    model: torch.nn.Module = CNNModel((128, 256), 3, 10)
    outputs: torch.Tensor = model(torch.rand(64, 3, 224, 224))

    # check object type
    assert isinstance(
        outputs, torch.Tensor
    ), f"Incorrect outputs type, expected tensor got {type(outputs)}"

    # check shape of output
    assert outputs.shape == (
        64,
        10,
    ), f"Incorrect outputs shape, expected (64, 10) got {outputs.shape}"

    return None
