# deep learning libraries
import torch
from torch.jit import RecursiveScriptModule

# own modules
from src.data import load_data
from src.utils import (
    Accuracy,
    load_model,
    set_seed,
)

from torch.utils.data import DataLoader
import numpy as np

# set device
device: torch.device = (
    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)

# set all seeds and set number of threads
set_seed(42)
torch.set_num_threads(8)

# static variables
DATA_PATH: str = "data"


accuracy_calculator = Accuracy()

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
            accuracy_calculator.reset()
            accuracy_calculator.update(logits=outputs, labels=targets)
            batch_accuracy = accuracy_calculator.compute()

            # Almacenar la precisión de este lote (batch)
            accuracies.append(batch_accuracy)

    # Calcular la precisión promedio para el conjunto de prueba
    avg_accuracy = float(np.mean(accuracies))

    # Devolver la precisión promedio
    return avg_accuracy

def main(name: str) -> float:
    """
    This function is the main program for the testing.
    """

    # TODO
    # load data
    test_data: DataLoader
    _, _, test_data =  load_data(DATA_PATH, batch_size=32,num_workers=8)


    # define model
    model: RecursiveScriptModule = torch.jit.load(f"models/{name}.pt").to(device)

    # call test step and evaluate accuracy
    accuracy: float = t_step(model, test_data, device)
  

    return accuracy


if __name__ == "__main__":
    print(f"accuracy: {main('best_model')}")
