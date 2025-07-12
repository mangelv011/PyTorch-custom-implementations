"""
This module contains the code to test the utils functions.
"""

# Standard libraries
import copy

# 3pps
import torch
from torch.optim.lr_scheduler import LRScheduler
import pytest

# own modules
from src.utils import StepLR, set_seed

# set seed
set_seed(42)


@pytest.mark.order(5)
def test_steplr() -> None:
    # define model
    model: torch.nn.Module = torch.nn.Sequential(
        torch.nn.Linear(30, 10), torch.nn.ReLU(), torch.nn.Linear(10, 1)
    )
    model_torch: torch.nn.Module = copy.deepcopy(model)

    # define inputs and targets
    inputs: torch.Tensor = torch.rand(64, 30)
    inputs_torch: torch.Tensor = inputs.clone().detach()
    targets: torch.Tensor = torch.rand(64, 1)

    # define loss and lr
    loss: torch.nn.Module = torch.nn.L1Loss()
    lr: float = 1e-3

    # define optimizers
    optimizer: torch.optim.Optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    optimizer_torch: torch.optim.Optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # define schedulers
    scheduler: LRScheduler = StepLR(optimizer, step_size=50, gamma=0.2)
    scheduler_torch: LRScheduler = torch.optim.lr_scheduler.StepLR(
        optimizer_torch, 50, gamma=0.2
    )

    # iter over epochs loop
    for epoch in range(110):
        # compute outputs for both models
        outputs: torch.Tensor = model(inputs)
        outputs_torch: torch.Tensor = model_torch(inputs_torch)

        # compute loss and optimize
        loss_value: torch.Tensor = loss(outputs, targets)
        optimizer.zero_grad()
        loss_value.backward()
        optimizer.step()

        # compute loss and optimize torch model
        loss_value = loss(outputs_torch, targets)
        optimizer_torch.zero_grad()
        loss_value.backward()
        optimizer_torch.step()

        # compute steps
        scheduler.step()
        scheduler_torch.step()

        # get lr and compare them
        lr = optimizer.param_groups[0]["lr"]
        lr_torch: float = optimizer_torch.param_groups[0]["lr"]
        print(lr)
        assert lr == lr_torch, (
            f"Incorrect step of scheduler, expected {lr_torch} in {epoch} epoch, "
            f"and got {lr}"
        )

    return None
