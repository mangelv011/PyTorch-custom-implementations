# deep learning libraries
import torch

# other libraries
import pytest

# own modules
from src.utils import set_seed, parameters_to_double
from src.models import RNN


@pytest.mark.order(1)
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


@pytest.mark.order(2)
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
