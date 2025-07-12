# deep learning libraries
import torch
import numpy as np  # noqa: F401
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# other libraries
from typing import Optional
from tqdm import tqdm


@torch.enable_grad()
def train_step(
    model: torch.nn.Module,
    train_data: DataLoader,
    mean: float,
    std: float,
    loss: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    writer: SummaryWriter,
    epoch: int,
    device: torch.device,
) -> None:
    """
    This function train the model.

    Args:
        model: model to train.
        train_data: dataloader of train data.
        mean: mean of the target.
        std: std of the target.
        loss: loss function.
        optimizer: optimizer.
        writer: writer for tensorboard.
        epoch: epoch of the training.
        device: device for running operations.
    """
    # Set model to training mode
    model.train()

    # Initialize metrics
    running_loss = 0.0
    running_mae = 0.0
    batch_count = 0

    # Iterate through batches
    for inputs, targets in tqdm(train_data):
        # Move data to device
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)

        # Safety check: ensure outputs are on the same device as targets
        if outputs.device != targets.device:
            outputs = outputs.to(targets.device, non_blocking=True)

        # Calculate loss using the provided loss function
        batch_loss = loss(outputs, targets)               
        # Calculate MAE on denormalized values
        denorm_outputs = outputs * std + mean
        denorm_targets = targets * std + mean
        mae = torch.mean(torch.abs(denorm_outputs - denorm_targets))

        # Backward pass
        batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Ensure all gradients are on the correct device before optimizer step
        for param in model.parameters():
            if param.grad is not None and param.grad.device != device:
                param.grad = param.grad.to(device, non_blocking=True)

        # Optimizer step
        optimizer.step()

        # Update metrics
        running_loss += batch_loss.item()
        running_mae += mae.item()
        batch_count += 1

    # Calculate average metrics
    avg_loss = running_loss / batch_count
    avg_mae = running_mae / batch_count

    # Log to tensorboard
    writer.add_scalar("Loss/train", avg_loss, epoch)
    writer.add_scalar("MAE/train", avg_mae, epoch)

    # Print progress
    print(f"Epoch {epoch} - Training Loss: {avg_loss:.6f}, MAE: {avg_mae:.6f}")


@torch.no_grad()
def val_step(
    model: torch.nn.Module,
    val_data: DataLoader,
    mean: float,
    std: float,
    loss: torch.nn.Module,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
    writer: SummaryWriter,
    epoch: int,
    device: torch.device,
) -> None:
    """
    This function train the model.

    Args:
        model: model to train.
        val_data: dataloader of validation data.
        mean: mean of the target.
        std: std of the target.
        loss: loss function.
        scheduler: scheduler.
        writer: writer for tensorboard.
        epoch: epoch of the training.
        device: device for running operations.
    """
    # Set model to evaluation mode
    model.eval()

    # Initialize metrics
    running_loss = 0.0
    running_mae = 0.0
    batch_count = 0

    # Iterate through batches
    for inputs, targets in val_data:
        # Move data to device
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Forward pass
        outputs = model(inputs)

        # Ensure outputs and targets are on the same device
        if outputs.device != targets.device:
            outputs = outputs.to(targets.device)

        # Calculate loss
        batch_loss = loss(outputs, targets)

        # Calculate MAE on denormalized values
        denorm_outputs = outputs * std + mean
        denorm_targets = targets * std + mean
        mae = torch.mean(torch.abs(denorm_outputs - denorm_targets))

        # Update metrics
        running_loss += batch_loss.item()
        running_mae += mae.item()
        batch_count += 1

    # Calculate average metrics
    avg_loss = running_loss / batch_count
    avg_mae = running_mae / batch_count

    # Log to tensorboard
    writer.add_scalar("Loss/validation", avg_loss, epoch)
    writer.add_scalar("MAE/validation", avg_mae, epoch)

    # Print progress
    print(f"Epoch {epoch} - Validation Loss: {avg_loss:.6f}, MAE: {avg_mae:.6f}")

    # Skip using the scheduler (but keep the parameter in the signature)
    # if scheduler is not None:
    #     scheduler.step(avg_loss)


@torch.no_grad()
def t_step(
    model: torch.nn.Module,
    test_data: DataLoader,
    mean: float,
    std: float,
    device: torch.device,
) -> float:
    """
    This function tests the model.

    Args:
        model: model to make predcitions.
        test_data: dataset for testing.
        mean: mean of the target.
        std: std of the target.
        device: device for running operations.

    Returns:
        mae of the test data.
    """
    # Set model to evaluation mode
    model.eval()

    # Initialize metrics
    total_mae = 0.0
    batch_count = 0

    # Iterate through batches
    for inputs, targets in test_data:
        # Move data to device
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Forward pass
        outputs = model(inputs)

        # Ensure outputs and targets are on the same device
        if outputs.device != targets.device:
            outputs = outputs.to(targets.device)

        # Denormalize predictions and targets
        denorm_outputs = outputs * std + mean
        denorm_targets = targets * std + mean

        # Calculate MAE
        mae = torch.mean(torch.abs(denorm_outputs - denorm_targets))

        # Update metrics
        total_mae += mae.item()
        batch_count += 1

    # Calculate final MAE
    avg_mae = total_mae / batch_count

    # Print result
    print(f"Test MAE: {avg_mae:.6f}", flush=True)

    return avg_mae
