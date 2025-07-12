# 3pps
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from torch.utils.data import DataLoader

# own modules
from src.models import CNNModel
from src.utils import (
    load_imagenette_data,
    Accuracy,
    save_model,
    set_seed,
)

# set device
device: torch.device = (
    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)

# set all seeds and set number of threads
set_seed(42)
torch.set_num_threads(8)

torch.backends.cudnn.benchmark = True


# static variables
DATA_PATH: str = "data"
NUMBER_OF_CLASSES: int = 10

accuracy_calculator = Accuracy()

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


    for inputs, targets in tqdm(train_data):
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
        accuracy_calculator.reset()
        accuracy_calculator.update(logits=outputs, labels=targets)
        batch_accuracy = accuracy_calculator.compute()

        # Almacenar la pérdida y la precisión de este lote (batch)
        losses.append(batch_loss.item())  # Convertir el tensor en un valor escalar

        accuracies.append(batch_accuracy)




    # write on tensorboard
    writer.add_scalar("train/loss", np.mean(losses), epoch) 
    writer.add_scalar("train/accuracy", np.mean(accuracies), epoch)  # Sin .cpu().item()



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
        for inputs, targets in tqdm(val_data):
            # Mover las entradas y los objetivos al dispositivo adecuado
            inputs, targets = inputs.to(device), targets.to(device)

            # Paso hacia adelante (forward pass)
            outputs = model(inputs)

            # Calcular la pérdida
            batch_loss = loss(outputs, targets)

            # Calcular la precisión
            accuracy_calculator.reset()
            accuracy_calculator.update(logits=outputs, labels=targets)
            batch_accuracy = accuracy_calculator.compute()

            # Almacenar la pérdida y la precisión de este lote (batch)
            losses.append(batch_loss.item())
            accuracies.append(batch_accuracy)

    # Registrar las métricas en TensorBoard
    writer.add_scalar("val/loss", np.mean(losses), epoch)
    writer.add_scalar("val/accuracy", np.mean(accuracies), epoch)  # Sin .cpu().item()



def main() -> None:
    """
    This function is the main program for the training.
    """

    # TODO
    hidden_sizes = (64,128)
    batch_size = 32
    lr = 1e-3
    epochs = 50
    dropout_prob = 0.25
    dataloader_workers = 8

    open("nohup.out", "w").close()

    # load data
    train_data: DataLoader
    val_data: DataLoader
    train_data, val_data, _ = load_imagenette_data(DATA_PATH, batch_size=batch_size, num_workers=dataloader_workers)

    # define name and writer
    name: str = f"model_lr_{lr}_hs_{hidden_sizes}_{batch_size}_{epochs}"
    writer: SummaryWriter = SummaryWriter(f"runs/{name}")

    # define model
  
    model: torch.nn.Module = CNNModel(hidden_sizes=hidden_sizes, output_channels=NUMBER_OF_CLASSES, dropout_prob=dropout_prob
       
    ).to(device)

    # define loss and optimizer
    loss: torch.nn.Module = torch.nn.CrossEntropyLoss()
    optimizer: torch.optim.Optimizer = torch.optim.Adam(model.parameters(), lr=lr)

     # train loop

    for epoch in range(epochs):
        print(f"Epoch: {epoch+1}: ")
        # call train step
        train_step(model, train_data, loss, optimizer, writer, epoch, device)


        # call val step
        val_step(model, val_data, loss, writer, epoch, device)


    # save model
    save_model(model, name)

    return None

if __name__ == "__main__":
    main()
