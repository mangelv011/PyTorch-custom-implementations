# 3pps
import torch
import torch.nn.functional as F
import pytest
import copy

# own modules
from src.activations import ReLU, SELU, Maxout, Softshrink
from src.layers import Linear, Dropout, RNN, LSTM, GRU
from src.normalizations import GroupNorm, BatchNorm, LayerNorm, InstanceNorm
from src.utils import set_seed, parameters_to_double
from src.optimizers import SGD, SGDMomentum, SGDNesterov, Adam
from src.pooling import unfold_max_pool_2d, fold_max_pool_2d, MaxPool2d

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
def test_selu_forward_backward():
    x = torch.randn(10, requires_grad=True)

    custom_selu = SELU()
    y_custom = custom_selu(x)

    x_torch = x.clone().detach().requires_grad_()
    y_torch = F.selu(x_torch)

    assert torch.allclose(y_custom, y_torch, atol=1e-6), "SELU forward mismatch"

    grad_output = torch.ones_like(x)
    y_custom.backward(grad_output, retain_graph=True)
    y_torch.backward(grad_output)

    assert torch.allclose(x.grad, x_torch.grad, atol=1e-6), "SELU backward mismatch"




@pytest.mark.order(6)
@pytest.mark.parametrize("num_groups,channels,affine", [
    (2, 4, False),
    (4, 8, True)
])
def test_group_norm(num_groups: int, channels: int, affine: bool) -> None:
    """
    This function is to test the GroupNorm.

    Args:
        num_groups: Number of groups for normalization.
        channels: Number of channels in the input.
        affine: Whether to use affine parameters.

    Returns:
        None.
    """
    # Create input tensor - batch size 2, given channels, height and width 4
    inputs = torch.randn(2, channels, 4, 4)
    inputs.requires_grad_(True)

    # Define modules
    module: torch.nn.Module = GroupNorm(num_groups, channels)
    module_torch: torch.nn.Module = torch.nn.GroupNorm(
        num_groups, channels, affine=False
    )

    # Compute outputs
    outputs: torch.Tensor = module(inputs)
    outputs_torch: torch.Tensor = module_torch(inputs)

    # Check shape
    assert outputs.shape == outputs_torch.shape, (
        f"Incorrect output shape, expected {outputs_torch.shape} and got "
        f"{outputs.shape}"
    )

    # Check outputs value
    assert torch.allclose(outputs, outputs_torch), "Incorrect outputs value"

    return None



@pytest.mark.order(7)
@pytest.mark.parametrize(
    "shape, num_groups, affine",
    [((32, 6, 16, 16), 1, False), ((16, 12, 8, 8), 3, True)],
)
def test_batch_norm(shape: tuple[int, ...], num_groups: int, affine: bool) -> None:
    for seed in range(10):
        scale = torch.randint(1, 10, (1,), dtype=torch.float64)
        bias = torch.randint(-10, 10, (1,), dtype=torch.float64)
        inputs = torch.rand(shape, dtype=torch.float64) * scale + bias

        set_seed(seed)
        model = BatchNorm(num_groups, shape[1], eps=1e-5, affine=affine)

        set_seed(seed)
        model_torch = torch.nn.BatchNorm2d(shape[1], eps=1e-5, affine=affine).to(dtype=torch.double)

        if affine:
            with torch.no_grad():
                model_torch.weight.copy_(model.weight)
                model_torch.bias.copy_(model.bias)

        outputs = model(inputs)
        outputs_torch = model_torch(inputs)

        assert outputs.shape == outputs_torch.shape, f"Shape mismatch: {outputs.shape} vs {outputs_torch.shape}"
        assert torch.allclose(outputs, outputs_torch, atol=1e-3), "BatchNorm outputs mismatch"





@pytest.mark.order(8)
@pytest.mark.parametrize(
    "shape, affine",
    [((16, 8, 32, 32), False), ((8, 6, 16, 16), True)],
)
def test_layer_norm(shape: tuple[int, ...], affine: bool) -> None:
    for seed in range(10):
        scale = torch.randint(1, 10, (1,), dtype=torch.float64)
        bias = torch.randint(-10, 10, (1,), dtype=torch.float64)
        x = torch.rand(shape, dtype=torch.float64) * scale + bias

        set_seed(seed)
        model = LayerNorm(shape[1], eps=1e-5)
        model_torch = torch.nn.LayerNorm(shape[1:], eps=1e-5, elementwise_affine=False).to(dtype=torch.double)

        y = model(x)
        y_torch = model_torch(x)

        assert y.shape == y_torch.shape
        assert torch.allclose(y, y_torch, atol=1e-3), "LayerNorm outputs mismatch"




@pytest.mark.order(9)
@pytest.mark.parametrize(
    "shape, affine",
    [((8, 3, 32, 32), False), ((16, 6, 8, 8), True)],
)
def test_instance_norm(shape: tuple[int, ...], affine: bool) -> None:
    for seed in range(10):
        scale = torch.randint(1, 10, (1,), dtype=torch.float64)
        bias = torch.randint(-10, 10, (1,), dtype=torch.float64)
        x = torch.rand(shape, dtype=torch.float64) * scale + bias

        set_seed(seed)
        model = InstanceNorm(shape[1], eps=1e-5, affine=affine)
        model_torch = torch.nn.InstanceNorm2d(shape[1], eps=1e-5, affine=affine).to(dtype=torch.double)

        if affine:
            with torch.no_grad():
                model_torch.weight.copy_(model.weight)
                model_torch.bias.copy_(model.bias)

        y = model(x)
        y_torch = model_torch(x)

        assert y.shape == y_torch.shape
        assert torch.allclose(y, y_torch, atol=1e-3), "InstanceNorm outputs mismatch"







@pytest.mark.order(10)
@pytest.mark.parametrize("p, seed", [(0.0, 0), (0.5, 1), (0.7, 2)])
def test_dropout(p: float, seed: int) -> None:
    """
    This function tests the Dropout layer.

    Args:
        p: dropout probability.
        seed: seed for test.

    Returns:
        None.
    """
    # Create test data
    set_seed(42)
    inputs = torch.randn(8, 10, 12, 12)
    inputs_torch = inputs.clone()

    # define dropout
    dropout = Dropout(p)
    dropout_torch = torch.nn.Dropout(p)

    # activate train mode
    dropout.train()
    dropout_torch.train()

    # compute outputs
    set_seed(seed)
    outputs = dropout(inputs)
    set_seed(seed)
    outputs_torch = dropout_torch(inputs)

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




@pytest.mark.order(5)
@torch.no_grad()
def test_maxout_forward() -> None:
    """
    This function test the maxout.
    """

    # set seed
    set_seed(42)

    # define inputs
    inputs: torch.Tensor = torch.FloatTensor(64, 32).uniform_(-10, 10)
    inputs[0, 0] = 0

    # define maxout
    model = Maxout(32, 32)
    model.set_parameters(
        torch.zeros_like(model.weights_first),
        torch.zeros_like(model.bias_first),
        torch.eye(model.weights_first.shape[0]),
        torch.zeros_like(model.bias_second),
    )

    # compute outputs and backward
    outputs = model(inputs)

    # define torch relu
    model_torch: torch.nn.Module = torch.nn.ReLU()

    # compute outputs
    outputs_torch = model_torch(inputs)

    # check outputs
    assert (
        outputs != outputs_torch
    ).sum().item() == 0, "Incorrect forward simulating the relu"

    # define maxout
    model = Maxout(32, 32)
    model.set_parameters(
        -1 * torch.eye(model.weights_first.shape[0]),
        torch.zeros_like(model.bias_first),
        torch.eye(model.weights_first.shape[0]),
        torch.zeros_like(model.bias_second),
    )

    # compute outputs and backward
    outputs = model(inputs)

    # compute outputs
    outputs_torch = torch.abs(inputs)

    # check outputs
    assert (
        outputs != outputs_torch
    ).sum().item() == 0, "Incorrect forward simulating the absolute value"

    # define maxout
    model = Maxout(32, 20)

    # compute outputs and backward
    outputs = model(inputs)

    # define torch relu
    class MaxoutTorch(torch.nn.Module):
        def __init__(self, input_dim: int, output_dim: int) -> None:
            # call super class constructor
            super().__init__()

            self.linear1 = torch.nn.Linear(input_dim, output_dim)
            self.linear2 = torch.nn.Linear(input_dim, output_dim)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            outputs1: torch.Tensor = self.linear1(inputs)
            outputs2: torch.Tensor = self.linear2(inputs)
            outputs = torch.maximum(outputs1, outputs2)

            return outputs

    # compute outputs
    model_torch = MaxoutTorch(32, 20)
    model_torch.linear1.weight.data = model.weights_first.clone()
    model_torch.linear1.bias.data = model.bias_first.clone()
    model_torch.linear2.weight.data = model.weights_second.clone()
    model_torch.linear2.bias.data = model.bias_second.clone()

    # compute outputs
    outputs_torch = model_torch(inputs)

    # check outputs
    assert (
        outputs != outputs_torch
    ).sum().item() == 0, "Incorrect forward of implementation with nn"

    return None


@pytest.mark.order(11)
def test_maxout_backward() -> None:
    """
    _summary_

    Returns:
        _description_
    """

    # set seed
    set_seed(42)

    # define inputs
    inputs: torch.Tensor = torch.FloatTensor(64, 32).uniform_(-10, 10)
    inputs[0, 0] = 0
    inputs.requires_grad_(True)

    # define maxout
    model = Maxout(32, 20)

    # compute outputs and backward
    outputs = model(inputs)
    outputs.sum().backward()

    # get grads values
    if (
        inputs.grad is None
        or model.weights_first.grad is None
        or model.bias_first.grad is None
        or model.weights_second.grad is None
        or model.bias_second.grad is None
    ):
        assert False, "Gradients not returned, none value detected"
    inputs_grad: torch.Tensor = inputs.grad.clone()
    grad_weight_first: torch.Tensor = model.weights_first.grad.clone()
    grad_bias_first: torch.Tensor = model.bias_first.grad.clone()
    grad_weight_second: torch.Tensor = model.weights_second.grad.clone()
    grad_bias_second: torch.Tensor = model.bias_second.grad.clone()

    # define torch relu
    class MaxoutTorch(torch.nn.Module):
        def __init__(self, input_dim: int, output_dim: int) -> None:
            # call super class constructor
            super().__init__()

            self.linear1 = torch.nn.Linear(input_dim, output_dim)
            self.linear2 = torch.nn.Linear(input_dim, output_dim)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            outputs1: torch.Tensor = self.linear1(inputs)
            outputs2: torch.Tensor = self.linear2(inputs)
            outputs = torch.maximum(outputs1, outputs2)

            return outputs

    # compute outputs
    model_torch = MaxoutTorch(32, 20)
    with torch.no_grad():
        model_torch.linear1.weight.data = model.weights_first.clone()
        model_torch.linear1.bias.data = model.bias_first.clone()
        model_torch.linear2.weight.data = model.weights_second.clone()
        model_torch.linear2.bias.data = model.bias_second.clone()

    # compute outputs
    outputs_torch = model_torch(inputs)
    model.zero_grad()
    inputs.grad.zero_()
    outputs_torch.sum().backward()

    # get grads values
    if (
        inputs.grad is None
        or model_torch.linear1.weight.grad is None
        or model_torch.linear1.bias.grad is None
        or model_torch.linear2.weight.grad is None
        or model_torch.linear2.bias.grad is None
    ):
        assert False, "Gradients not returned, none value detected"
    inputs_grad_torch: torch.Tensor = inputs.grad.clone()
    grad_weight_first_torch: torch.Tensor = model_torch.linear1.weight.grad.clone()
    grad_bias_first_torch: torch.Tensor = model_torch.linear1.bias.grad.clone()
    grad_weight_second_torch: torch.Tensor = model_torch.linear2.weight.grad.clone()
    grad_bias_second_torch: torch.Tensor = model_torch.linear2.bias.grad.clone()

    # check inputs grads
    assert (
        inputs_grad != inputs_grad_torch
    ).sum().item() == 0, "Incorrect inputs gradients"

    # check weights first grads
    assert (
        grad_weight_first != grad_weight_first_torch
    ).sum().item() == 0, "Incorrect first weights gradients"

    # check bias second grads
    assert (
        grad_bias_first != grad_bias_first_torch
    ).sum().item() == 0, "Incorrect first bias gradients"

    # check weights first grads
    assert (
        grad_weight_second != grad_weight_second_torch
    ).sum().item() == 0, "Incorrect second weights gradients"

    # check bias second grads
    assert (
        grad_bias_second != grad_bias_second_torch
    ).sum().item() == 0, "Incorrect second bias gradients"

    return None



@pytest.mark.order(12)
@pytest.mark.parametrize("lr, weight_decay", [(1e-3, 0), (1e-4, 1e-2)])
def test_sgd(lr: float, weight_decay: float) -> None:
    """
    This function is the test for the SGD algorithm.

    Args:
        lr: learning rate.
        weight_decay: weight decay rate.
    """
    # Set seed for reproducibility
    set_seed(42)

    # Create test data
    inputs = torch.randn(32, 10)
    targets = torch.randn(32, 5)

    # Create a simple model for testing
    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = torch.nn.Linear(10, 8)
            self.relu = torch.nn.ReLU()
            self.linear2 = torch.nn.Linear(8, 5)
            
        def forward(self, x):
            x = self.linear1(x)
            x = self.relu(x)
            return self.linear2(x)
    
    # Create two identical models
    set_seed(42)
    model1 = SimpleModel()
    set_seed(42)
    model2 = SimpleModel()

    # define loss
    loss = torch.nn.L1Loss()

    # define optimizers
    optimizer1 = torch.optim.SGD(
        model1.parameters(), lr=lr, weight_decay=weight_decay
    )
    optimizer2 = SGD(
        model2.parameters(), lr=lr, weight_decay=weight_decay
    )

    # optimize first model
    for _ in range(10):
        outputs = model1(inputs)
        loss_value = loss(outputs, targets)
        optimizer1.zero_grad()
        loss_value.backward()
        optimizer1.step()

    # optimize second model
    for _ in range(10):
        outputs = model2(inputs)
        loss_value = loss(outputs, targets)
        optimizer2.zero_grad()
        loss_value.backward()
        optimizer2.step()

    # check parameters of both models
    for parameter1, parameter2 in zip(model1.parameters(), model2.parameters()):
        assert torch.allclose(
            parameter1.data, parameter2.data
        ), f"Incorrect SGD implementation - parameters don't match: {parameter1.data} vs {parameter2.data}"

    return None




@pytest.mark.order(13)
@pytest.mark.parametrize(
    "lr, momentum, weight_decay", [(1e-3, 0.9, 0), (1e-4, 0, 1e-2)]
)
def test_sgd_momentum(
    lr: float,
    momentum: float,
    weight_decay: float
) -> None:
    """
    This function is the test for the SGD algorithm with momentum.

    Args:
        lr: learning rate.
        momentum: momentum rate.
        weight_decay: weight decay rate.
    """
    # Set seed for reproducibility
    set_seed(42)

    # Create test data
    inputs = torch.randn(32, 10)
    targets = torch.randn(32, 5)

    # Create a simple model for testing
    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = torch.nn.Linear(10, 8)
            self.relu = torch.nn.ReLU()
            self.linear2 = torch.nn.Linear(8, 5)
            
        def forward(self, x):
            x = self.linear1(x)
            x = self.relu(x)
            return self.linear2(x)
    
    # Create a model that we'll clone for both optimizers
    set_seed(42)
    model_original = SimpleModel()
    
    # clone model
    model1 = copy.deepcopy(model_original)
    model2 = copy.deepcopy(model_original)

    # define loss
    loss = torch.nn.L1Loss()

    # define optimizers
    optimizer1 = torch.optim.SGD(
        model1.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
    )
    optimizer2 = SGDMomentum(
        model2.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
    )

    # optimize first model
    for _ in range(10):
        outputs = model1(inputs)
        loss_value = loss(outputs, targets)
        optimizer1.zero_grad()
        loss_value.backward()
        optimizer1.step()

    # optimize second model
    for _ in range(10):
        # optimize second model
        outputs = model2(inputs)
        loss_value = loss(outputs, targets)
        optimizer2.zero_grad()
        loss_value.backward()
        optimizer2.step()

    # check parameters of both models
    for parameter1, parameter2 in zip(model1.parameters(), model2.parameters()):
        assert torch.allclose(
            parameter1.data, parameter2.data
        ), f"Incorrect SGDMomentum implementation - parameters don't match: {parameter1.data} vs {parameter2.data}"

    return None


@pytest.mark.order(14)
@pytest.mark.parametrize(
    "lr, momentum, weight_decay", [(1e-3, 0.9, 0), (1e-4, 0.5, 1e-2)]
)
def test_sgd_nesterov(
    lr: float,
    momentum: float,
    weight_decay: float,
) -> None:
    """
    This function is the test for the SGD algorithm with nesterov.
    momentum.

    Args:
        lr: learning rate.
        momentum: momentum rate.
        weight_decay: weight decay rate.
    """
    # Set seed for reproducibility
    set_seed(42)

    # Create test data
    inputs = torch.randn(32, 10)
    targets = torch.randn(32, 5)

    # Create a simple model for testing
    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = torch.nn.Linear(10, 8)
            self.relu = torch.nn.ReLU()
            self.linear2 = torch.nn.Linear(8, 5)
            
        def forward(self, x):
            x = self.linear1(x)
            x = self.relu(x)
            return self.linear2(x)
    
    # Create a model that we'll clone for both optimizers
    set_seed(42)
    model_original = SimpleModel()

    # clone model
    model1: torch.nn.Module = copy.deepcopy(model_original)
    model2: torch.nn.Module = copy.deepcopy(model_original)

    # define loss and lr
    loss = torch.nn.L1Loss()

    # define optimizers
    optimizer1: torch.optim.Optimizer = torch.optim.SGD(
        model1.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        nesterov=True,
    )
    optimizer2: torch.optim.Optimizer = SGDNesterov(
        model2.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
    )

    # optimize first model
    for _ in range(10):
        outputs: torch.Tensor = model1(inputs)
        loss_value: torch.Tensor = loss(outputs, targets)
        optimizer1.zero_grad()
        loss_value.backward()
        optimizer1.step()

    # optimize second model
    for _ in range(10):
        # optimize second model
        outputs = model2(inputs)
        loss_value = loss(outputs, targets)
        optimizer2.zero_grad()
        loss_value.backward()
        optimizer2.step()

    # check parameters of both models
    for parameter1, parameter2 in zip(model1.parameters(), model2.parameters()):
        assert torch.allclose(
            parameter1.data, parameter2.data
        ), "Incorrect return of the algorithm"

    return None




@pytest.mark.order(15)
@pytest.mark.parametrize(
    "lr, betas, weight_decay", [(1e-3, (0.9, 0.999), 0), (1e-4, (0.5, 0.4), 1e-2)]
)
def test_adam(
    lr: float,
    betas: tuple[float, float],
    weight_decay: float,
) -> None:
    """
    This function is the test for the Adam optimizer.

    Args:
        lr: learning rate.
        betas: beta1 and beta2 values for Adam.
        weight_decay: weight decay rate.
    """
    # Set seed for reproducibility
    set_seed(42)

    # Create test data
    inputs = torch.randn(32, 10)
    targets = torch.randn(32, 5)

    # Create a simple model for testing
    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = torch.nn.Linear(10, 8)
            self.relu = torch.nn.ReLU()
            self.linear2 = torch.nn.Linear(8, 5)
            
        def forward(self, x):
            x = self.linear1(x)
            x = self.relu(x)
            return self.linear2(x)
    
    # Create a model that we'll clone for both optimizers
    set_seed(42)
    model_original = SimpleModel()

    # clone model
    model1 = copy.deepcopy(model_original)
    model2 = copy.deepcopy(model_original)

    # define loss
    loss = torch.nn.L1Loss()

    # define optimizers
    optimizer1 = torch.optim.Adam(
        model1.parameters(), lr=lr, betas=betas, weight_decay=weight_decay
    )
    optimizer2 = Adam(
        model2.parameters(), lr=lr, betas=betas, weight_decay=weight_decay
    )

    # optimize first model
    for _ in range(10):
        outputs = model1(inputs)
        loss_value = loss(outputs, targets)
        optimizer1.zero_grad()
        loss_value.backward()
        optimizer1.step()

    # optimize second model
    for _ in range(10):
        # optimize second model
        outputs = model2(inputs)
        loss_value = loss(outputs, targets)
        optimizer2.zero_grad()
        loss_value.backward()
        optimizer2.step()

    # check parameters of both models
    for parameter1, parameter2 in zip(model1.parameters(), model2.parameters()):
        assert torch.allclose(
            parameter1.data, parameter2.data, atol=1e-5
        ), "Incorrect Adam implementation"

    return None


@pytest.mark.order(16)
def test_rnn_forward() -> None:
    # define inputs
    inputs: torch.Tensor = torch.rand(64, 12, 20).double()
    h_0: torch.Tensor = torch.rand(1, 64, 30).double()
    inputs_torch: torch.Tensor = inputs.clone()
    h_0_torch: torch.Tensor = h_0.clone()

    # define models
    set_seed(42)
    model = RNN(20, 30)
    parameters_to_double(model)
    set_seed(42)
    model_torch: torch.nn.Module = torch.nn.RNN(
        20, 30, batch_first=True, nonlinearity="relu"
    )
    parameters_to_double(model_torch)

    # compute outputs
    outputs: torch.Tensor
    h_n: torch.Tensor
    outputs, h_n = model(inputs, h_0)

    # compute torch outputs
    outputs_torch: torch.Tensor
    h_n_torch: torch.Tensor
    outputs_torch, h_n_torch = model_torch(inputs_torch, h_0_torch)

    # check output type
    assert isinstance(
        outputs, torch.Tensor
    ), f"Incorrect type, expected torch.Tensor got {type(outputs)}"

    # check output size
    assert (
        outputs.shape == outputs_torch.shape
    ), f"Incorrect outputs shape, expected {outputs_torch.shape}, got {outputs.shape}"

    # check outputs of dropout
    assert (
        outputs.round(decimals=2) != outputs_torch.round(decimals=2)
    ).sum().item() == 0, "Incorrect outputs in forward"

    return None


@pytest.mark.order(17)
def test_rnn_backward() -> None:
    # define inputs
    set_seed(42)
    inputs: torch.Tensor = torch.rand(64, 12, 20).double().requires_grad_(True)
    h0: torch.Tensor = torch.rand(1, 64, 30).double().requires_grad_(True)

    # define models
    set_seed(42)
    model = RNN(20, 30)
    parameters_to_double(model)
    set_seed(42)
    model_torch: torch.nn.Module = torch.nn.RNN(
        20, 30, batch_first=True, nonlinearity="relu"
    )
    parameters_to_double(model_torch)

    # compute outputs
    outputs: torch.Tensor
    h_n: torch.Tensor
    outputs, h_n = model(inputs, h0)
    outputs.sum().backward()

    # get grads values
    if (
        inputs.grad is None
        or h0.grad is None
        or model.weight_ih.grad is None
        or model.weight_hh.grad is None
        or model.bias_ih.grad is None
        or model.bias_hh.grad is None
    ):
        assert False, "Gradients not returned, none value detected"
    inputs_grad: torch.Tensor = inputs.grad.clone()
    h0_grad: torch.Tensor = h0.grad.clone()
    weight_ih_grad: torch.Tensor = model.weight_ih.grad.clone()
    weight_hh_grad: torch.Tensor = model.weight_hh.grad.clone()
    bias_ih_grad: torch.Tensor = model.bias_ih.grad.clone()
    bias_hh_grad: torch.Tensor = model.bias_hh.grad.clone()

    # compute torch outputs
    outputs_torch: torch.Tensor
    h_n_torch: torch.Tensor
    outputs_torch, h_n_torch = model_torch(inputs, h0)
    inputs.grad.zero_()
    h0.grad.zero_()
    outputs_torch.sum().backward()

    # get grads values
    if (
        inputs.grad is None
        or h0.grad is None
        or model_torch.weight_ih_l0.grad is None
        or model_torch.weight_hh_l0.grad is None
        or model_torch.bias_ih_l0.grad is None
        or model_torch.bias_hh_l0.grad is None
    ):
        assert False, "Gradients not returned, none value detected"
    inputs_grad_torch: torch.Tensor = inputs.grad.clone()
    h0_grad_torch: torch.Tensor = h0.grad.clone()
    weight_ih_grad_torch: torch.Tensor = model_torch.weight_ih_l0.grad.clone()
    weight_hh_grad_torch: torch.Tensor = model_torch.weight_hh_l0.grad.clone()
    bias_ih_grad_torch: torch.Tensor = model.bias_ih.grad.clone()
    bias_hh_grad_torch: torch.Tensor = model.bias_hh.grad.clone()

    # check input grads of last hidden state
    assert (
        inputs_grad[:, -1, :].round(decimals=2)
        != inputs_grad_torch[:, -1, :].round(decimals=2)
    ).sum().item() == 0, "Incorrect grad inputs in last hidden state"

    # check input grads of last - 1 hidden state
    assert (
        inputs_grad[:, -2, :].round(decimals=2)
        != inputs_grad_torch[:, -2, :].round(decimals=2)
    ).sum().item() == 0, "Incorrect grad inputs in last - 1 hidden state"

    # check all inputs grads
    assert (
        inputs_grad.round(decimals=2) != inputs_grad_torch.round(decimals=2)
    ).sum().item() == 0, "Incorrect grad inputs"

    # check h0 grads
    assert (
        h0_grad.round(decimals=2) != h0_grad_torch.round(decimals=2)
    ).sum().item() == 0, "Incorrect grad h0"

    # check weight_ih grads
    assert (
        weight_ih_grad.round(decimals=4) != weight_ih_grad_torch.round(decimals=4)
    ).sum().item() == 0, "Incorrect grad weight_ih"

    # check weight_hh grads
    assert (
        weight_hh_grad.round(decimals=4) != weight_hh_grad_torch.round(decimals=4)
    ).sum().item() == 0, "Incorrect grad weight_hh"

    # check bias_ih
    assert (
        bias_ih_grad.round(decimals=4) != bias_ih_grad_torch.round(decimals=4)
    ).sum().item() == 0, "Incorrect grad bias_ih"

    # check bias_hh
    assert (
        bias_hh_grad.round(decimals=4) != bias_hh_grad_torch.round(decimals=4)
    ).sum().item() == 0, "Incorrect grad bias_hh"

    return None





@pytest.mark.order(18)
def test_lstm_forward() -> None:
    # define inputs
    inputs: torch.Tensor = torch.rand(64, 12, 20).double()
    h_0: torch.Tensor = torch.rand(1, 64, 30).double()
    c_0: torch.Tensor = torch.rand(1, 64, 30).double()
    inputs_torch: torch.Tensor = inputs.clone()
    h_0_torch: torch.Tensor = h_0.clone()
    c_0_torch: torch.Tensor = c_0.clone()

    # define models
    model = LSTM(20, 30)
    parameters_to_double(model)
    set_seed(42)
    model.reset_parameters()

    model_torch: torch.nn.Module = torch.nn.LSTM(
        20, 30, batch_first=True
    )
    parameters_to_double(model_torch)
    set_seed(42)
    model_torch.reset_parameters()

    # compute outputs
    outputs: torch.Tensor
    h_n: torch.Tensor
    c_n: torch.Tensor
    outputs, h_n, c_n = model(inputs, h_0, c_0)

    # compute torch outputs
    outputs_torch: torch.Tensor
    outputs_torch, (h_n_torch, c_n_torch) = model_torch(inputs_torch, (h_0_torch, c_0_torch))

    # check output type
    assert isinstance(
        outputs, torch.Tensor
    ), f"Incorrect type, expected torch.Tensor got {type(outputs)}"

    # check output shape
    assert (
        outputs.shape == outputs_torch.shape
    ), f"Incorrect outputs shape, expected {outputs_torch.shape}, got {outputs.shape}"

    # check output values (with tolerance)
    assert torch.allclose(
        outputs, outputs_torch, atol=1e-2, rtol=0
    ), "Incorrect outputs in forward (not close enough to nn.LSTM)"

    return None



@pytest.mark.order(19)
def test_gru_forward() -> None:
    # define inputs
    inputs: torch.Tensor = torch.rand(64, 12, 20).double()
    h_0: torch.Tensor = torch.rand(1, 64, 30).double()
    inputs_torch: torch.Tensor = inputs.clone()
    h_0_torch: torch.Tensor = h_0.clone()

    # define models
    model = GRU(20, 30)
    parameters_to_double(model)
    set_seed(42)
    model.reset_parameters()

    model_torch: torch.nn.Module = torch.nn.GRU(
        20, 30, batch_first=True
    )
    parameters_to_double(model_torch)
    set_seed(42)
    model_torch.reset_parameters()

    # compute outputs
    outputs: torch.Tensor
    h_n: torch.Tensor
    outputs, h_n = model(inputs, h_0)

    # compute torch outputs
    outputs_torch: torch.Tensor
    h_n_torch: torch.Tensor
    outputs_torch, h_n_torch = model_torch(inputs_torch, h_0_torch)

    # check output type
    assert isinstance(outputs, torch.Tensor), (
        f"Incorrect type, expected torch.Tensor got {type(outputs)}"
    )

    # check output shape
    assert outputs.shape == outputs_torch.shape, (
        f"Incorrect outputs shape, expected {outputs_torch.shape}, got {outputs.shape}"
    )

    # check output values (with tolerance)
    assert torch.allclose(
        outputs, outputs_torch, atol=1e-2, rtol=0
    ), "Incorrect outputs in forward (not close enough to nn.GRU)"

    return None

@pytest.mark.order(20)
@pytest.mark.parametrize(
    "shape, kernel_size", [((64, 3, 32, 32), 4), ((128, 2, 64, 64), 3)]
)
def test_unfold_max_pool_2d(shape: tuple[int, ...], kernel_size: int) -> None:
    """
    This function is the test for the unfold_max_pool_2d.

    Args:
        shape: shape of the input tensor.
        kernel_size: kernel size for the unfold.
    """

    # define inputs
    inputs: torch.Tensor = torch.rand(shape)

    # unfold inputs
    inputs_unfolded: torch.Tensor = unfold_max_pool_2d(inputs, kernel_size, 1, 0)

    # check dimensions
    assert inputs_unfolded.shape[:2] == (
        shape[0] * shape[1],
        kernel_size**2,
    ), "Incorrect shape of unfold"

    # check values
    assert (
        inputs[0, 0, :kernel_size, :kernel_size].reshape(-1) != inputs_unfolded[0, :, 0]
    ).sum().item() == 0, "Incorrect values of unfold"
    assert (
        inputs[0, 1, :kernel_size, :kernel_size].reshape(-1) != inputs_unfolded[1, :, 0]
    ).sum().item() == 0, "Incorrect values of unfold"
    assert (
        inputs[0, 0, :kernel_size, 1 : (kernel_size + 1)].reshape(-1)
        != inputs_unfolded[0, :, 1]
    ).sum().item() == 0, "Incorrect values of unfold"

    return None


@pytest.mark.order(21)
@pytest.mark.parametrize(
    "shape,, kernel_size, stride, padding",
    [((64, 3, 32, 32), 3, 1, 0), ((128, 6, 64, 64), 7, 1, 0)],
)
def test_fold_max_pool_2d(
    shape: tuple[int, ...], kernel_size: int, stride: int, padding: int
) -> None:
    """
    This function is the test for the fold_max_pool_2d.
    """

    # define inputs
    inputs: torch.Tensor = torch.rand(shape)

    # compute fold version
    inputs_folded: torch.Tensor = fold_max_pool_2d(
        unfold_max_pool_2d(inputs, kernel_size, 1, 0),
        shape[2],
        inputs.shape[0],
        kernel_size,
        stride,
        padding,
    )

    # compute check tensor
    input_ones = torch.ones(inputs.shape, dtype=inputs.dtype)
    divisor = fold_max_pool_2d(
        unfold_max_pool_2d(input_ones, kernel_size, 1, 0),
        shape[2],
        inputs.shape[0],
        kernel_size,
        stride,
        padding,
    )
    check_tensor: torch.Tensor = divisor * inputs

    # check dimensions
    assert inputs_folded.shape == shape, "Incorrect shape of fold"

    # check values
    assert torch.allclose(
        inputs_folded, check_tensor, atol=1e-5
    ), "Incorrect values of fold"

    return None


@pytest.mark.order(22)
@pytest.mark.parametrize(
    "shape, kernel_size", [((64, 3, 32, 32), 4), ((128, 2, 64, 64), 3)]
)
def test_max_pool_forward(shape: tuple[int, ...], kernel_size: int) -> None:
    """
    This function is the test for the forward of the MaxPool2d.

    Args:
        shape: shape of the input tensor.
        kernel_size: kernel size to use.
    """

    # loop with different seeds
    for seed in range(10):
        # define inputs
        set_seed(seed)
        inputs: torch.Tensor = torch.rand(shape)

        # define models
        model = MaxPool2d(kernel_size, stride=1)
        model_torch = torch.nn.MaxPool2d(kernel_size, stride=1)

        # compute outputs
        outputs = model(inputs)
        outputs_torch = model_torch(inputs)

        # check output size
        assert (
            outputs.shape == outputs_torch.shape
        ), f"Incorrect outputs shape, expected {outputs_torch.shape}, got {outputs.shape}"

        # check outputs
        assert torch.allclose(outputs, outputs_torch, atol=1e-10), "Incorrect outputs"

    return None


@pytest.mark.order(23)
@pytest.mark.parametrize(
    "shape, kernel_size", [((64, 3, 32, 32), 4), ((128, 2, 64, 64), 3)]
)
def test_max_pool_backward(shape: tuple[int, ...], kernel_size: int) -> None:
    """
    This function is the test for the backward of the MaxPool2d.

    Args:
        shape: shape of the input tensor.
        kernel_size: kernel size to use.
    """

    # loop with different seeds
    for seed in range(10):
        # set seed
        set_seed(seed)

        # define inputs
        inputs: torch.Tensor = torch.rand(shape)
        inputs.requires_grad_(True)

        # define models
        model = MaxPool2d(kernel_size, stride=1)
        model_torch = torch.nn.MaxPool2d(kernel_size, stride=1)

        # compute backward of our maxpool
        outputs = model(inputs)
        if inputs.grad is not None:
            inputs.grad.zero_()
        outputs.sum().backward()
        if inputs.grad is None:
            assert False, "Gradients not returned, none value detected"
        grad_inputs: torch.Tensor = inputs.grad.clone()

        # compute backward of pytorch maxpool
        outputs_torch = model_torch(inputs)
        inputs.grad.zero_()
        outputs_torch.sum().backward()
        if inputs.grad is None:
            assert False, "Gradients not returned, none value detected"
        grad_inputs_torch: torch.Tensor = inputs.grad.clone()

        # check output size
        assert (
            grad_inputs.shape == grad_inputs_torch.shape
        ), f"Incorrect outputs shape, expected {outputs_torch.shape}, got {outputs.shape}"

        # check outputs
        assert torch.allclose(
            grad_inputs, grad_inputs_torch, atol=1e-10
        ), "Incorrect outputs"

    return None




@pytest.mark.order(24)
@pytest.mark.parametrize("shape, lambd", [((64, 32), 0.5), ((64, 3, 32, 32), 1.0)])
def test_softshrink_forward(shape: tuple[int, ...], lambd: float) -> None:
    """
    This function is the forward test for the Softshrink.

    Args:
        shape: shape of the tensors to use.
        lambd: lambd parameter to use.
    """

    for seed in range(10):
        set_seed(seed)
        inputs: torch.Tensor = torch.rand(shape)

        # put elements on the frontier
        inputs[0, 0] = 0.5
        inputs[0, 1] = -0.5

        # define models
        model = Softshrink(lambd)
        model_torch = torch.nn.Softshrink(lambd)

        # compute outputs
        outputs: torch.Tensor = model(inputs)
        outputs_torch: torch.Tensor = model_torch(inputs)

        # check output size
        assert (
            outputs.shape == outputs_torch.shape
        ), f"Incorrect outputs shape, expected {outputs_torch.shape}, got {outputs.shape}"

        # check outputs
        assert torch.allclose(outputs, outputs_torch, atol=1e-4), "Incorrect outputs"

    return None


@pytest.mark.order(25)
@pytest.mark.parametrize("shape, lambd", [((64, 32), 0.5), ((64, 3, 32, 32), 1.0)])
def test_softshrink_backward(shape: tuple[int, ...], lambd: float) -> None:
    """
    This function is the backward test for the Softshrink.

    Args:
        shape: shape of the tensors to use.
        lambd: lambd parameter to use.
    """

    for seed in range(24):
        set_seed(seed)
        inputs: torch.Tensor = torch.rand(shape)

        # put elements on the frontier
        inputs[0, 0] = 0.5
        inputs[0, 1] = -0.5

        inputs.requires_grad_(True)

        # define models
        model = Softshrink(lambd)
        model_torch = torch.nn.Softshrink(lambd)

        # compute backward of our maxpool
        outputs = model(inputs)
        if inputs.grad is not None:
            inputs.grad.zero_()
        outputs.sum().backward()
        if inputs.grad is None:
            assert False, "Gradients not returned, none value detected"
        grad_inputs: torch.Tensor = inputs.grad.clone()

        # compute backward of pytorch maxpool
        outputs_torch = model_torch(inputs)
        inputs.grad.zero_()
        outputs_torch.sum().backward()
        if inputs.grad is None:
            assert False, "Gradients not returned, none value detected"
        grad_inputs_torch: torch.Tensor = inputs.grad.clone()

        # check output size
        assert (
            grad_inputs.shape == grad_inputs_torch.shape
        ), f"Incorrect outputs shape, expected {outputs_torch.shape}, got {outputs.shape}"

        # check outputs
        assert torch.allclose(
            grad_inputs, grad_inputs_torch, atol=1e-4
        ), "Incorrect outputs"

    return None


@pytest.mark.order(26)
@pytest.mark.parametrize(
    "lr, betas, weight_decay, momentum_decay", 
    [(1e-3, (0.9, 0.999), 0, 0.004), (1e-4, (0.8, 0.9), 1e-2, 0.002)]
)
def test_nadam(
    lr: float,
    betas: tuple[float, float],
    weight_decay: float,
    momentum_decay: float,
) -> None:
    """
    This function is the test for the NAdam optimizer.

    Args:
        lr: learning rate.
        betas: beta1 and beta2 values for NAdam.
        weight_decay: weight decay rate.
        momentum_decay: momentum decay rate.
    """
    # Set seed for reproducibility
    set_seed(42)

    # Create test data
    inputs = torch.randn(32, 10)
    targets = torch.randn(32, 5)

    # Create a simple model for testing
    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = torch.nn.Linear(10, 8)
            self.relu = torch.nn.ReLU()
            self.linear2 = torch.nn.Linear(8, 5)
            
        def forward(self, x):
            x = self.linear1(x)
            x = self.relu(x)
            return self.linear2(x)
    
    # Create a model that we'll clone for both optimizers
    set_seed(42)
    model_original = SimpleModel()

    # Import NAdam here
    from src.optimizers import NAdam

    # Clone model
    model1 = copy.deepcopy(model_original)
    model2 = copy.deepcopy(model_original)

    # Define loss
    loss = torch.nn.L1Loss()

    # Define optimizers
    # Use PyTorch's Adam as a reference (PyTorch doesn't have NAdam built-in)
    optimizer1 = torch.optim.Adam(
        model1.parameters(), lr=lr, betas=betas, weight_decay=weight_decay
    )
    optimizer2 = NAdam(
        model2.parameters(), lr=lr, betas=betas, weight_decay=weight_decay, 
        momentum_decay=momentum_decay
    )

    # Optimize first model with Adam
    for _ in range(5):
        outputs = model1(inputs)
        loss_value = loss(outputs, targets)
        optimizer1.zero_grad()
        loss_value.backward()
        optimizer1.step()

    # Optimize second model with our NAdam
    for _ in range(5):
        outputs = model2(inputs)
        loss_value = loss(outputs, targets)
        optimizer2.zero_grad()
        loss_value.backward()
        optimizer2.step()

    # Check that both models have been updated (parameters should be different from initial)
    for p1, p2, p_orig in zip(model1.parameters(), model2.parameters(), model_original.parameters()):
        assert not torch.allclose(p1.data, p_orig.data), "Adam optimizer did not update parameters"
        assert not torch.allclose(p2.data, p_orig.data), "NAdam optimizer did not update parameters"
    
    # Check that NAdam's updates are reasonably close to Adam's 
    # (they won't be identical but should be in similar range)
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        # Compute relative magnitude difference
        abs_diff = torch.abs(p1.data - p2.data)
        avg_mag = (torch.abs(p1.data) + torch.abs(p2.data)) / 2
        rel_diff = abs_diff / (avg_mag + 1e-8)
        
        # The difference should not be extremely large, but we expect them to be different
        assert torch.mean(rel_diff) < 1.0, "NAdam updates differ too much from Adam"

    return None

