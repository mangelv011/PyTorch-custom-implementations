# deep learning libraries
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# own modules
from src.utils import accuracy


def train_step(
    model: torch.nn.Module,
    train_data: DataLoader,
    loss: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    writer: SummaryWriter,
    epoch: int,
    device: torch.device,
) -> None:
    """
    This function computes the training step.

    Args:
        model: pytorch model.
        train_data: train dataloader.
        loss: loss function.
        optimizer: optimizer object.
        writer: tensorboard writer.
        epoch: epoch number.
        device: device of model.
    """

    # define metric lists
    losses: list[float] = []
    accuracies: list[float] = []

    model.train()  # Poner el modelo en modo de entrenamiento

    for inputs, targets in train_data:
        # Mover las entradas y los objetivos al dispositivo adecuado (GPU o CPU)
        inputs, targets = inputs.to(device), targets.to(device)

        # Inicializar los gradientes a cero
        optimizer.zero_grad()

        # Paso hacia adelante (forward pass)
        outputs = model(inputs)

        # Calcular la pérdida
        batch_loss = loss(outputs, targets)

        # Retropropagación (backpropagation)
        batch_loss.backward()  # definida en la clase base
        optimizer.step()

        # Calcular la precisión
        batch_accuracy = accuracy(outputs, targets)

        # Almacenar la pérdida y la precisión de este lote (batch)
        losses.append(batch_loss.item())
        accuracies.append(batch_accuracy.item())

    # write on tensorboard
    writer.add_scalar("train/loss", np.mean(losses), epoch)
    writer.add_scalar("train/accuracy", np.mean(accuracies), epoch)


def val_step(
    model: torch.nn.Module,
    val_data: DataLoader,
    loss: torch.nn.Module,
    writer: SummaryWriter,
    epoch: int,
    device: torch.device,
) -> None:
    """
    This function computes the validation step.

    Args:
        model: pytorch model.
        val_data: dataloader of validation data.
        loss: loss function.
        writer: tensorboard writer.
        epoch: epoch number.
        device: device of model.
    """

    # Definir listas para almacenar las métricas
    losses: list[float] = []
    accuracies: list[float] = []

    model.eval()  # Poner el modelo en modo de evaluación

    # Practicamente igual, pero no se consideran los gradientes
    with torch.no_grad():
        for inputs, targets in val_data:
            # Mover las entradas y los objetivos al dispositivo adecuado
            inputs, targets = inputs.to(device), targets.to(device)

            # Paso hacia adelante (forward pass)
            outputs = model(inputs)

            # Calcular la pérdida
            batch_loss = loss(outputs, targets)

            # Calcular la precisión
            batch_accuracy = accuracy(outputs, targets)

            # Almacenar la pérdida y la precisión de este lote (batch)
            losses.append(batch_loss.item())
            accuracies.append(batch_accuracy.item())

    # Registrar las métricas en TensorBoard
    writer.add_scalar("val/loss", np.mean(losses), epoch)
    writer.add_scalar("val/accuracy", np.mean(accuracies), epoch)


def t_step(
    model: torch.nn.Module,
    test_data: DataLoader,
    device: torch.device,
) -> float:
    """
    This function computes the test step.

    Args:
        model: pytorch model.
        val_data: dataloader of test data.
        device: device of model.

    Returns:
        average accuracy.
    """

    # Definir una lista para almacenar las precisiones
    accuracies: list[float] = []

    model.eval()  # Poner el modelo en modo de evaluación

    # No calcular los gradientes durante la prueba
    with torch.no_grad():
        for inputs, targets in test_data:
            # Mover las entradas y los objetivos al dispositivo adecuado
            inputs, targets = inputs.to(device), targets.to(device)

            # Paso hacia adelante (forward pass)
            outputs = model(inputs)

            # Calcular la precisión
            batch_accuracy = accuracy(outputs, targets)

            # Almacenar la precisión de este lote (batch)
            accuracies.append(batch_accuracy.item())

    # Calcular la precisión promedio para el conjunto de prueba
    avg_accuracy = np.mean(accuracies)

    # Devolver la precisión promedio
    return avg_accuracy
