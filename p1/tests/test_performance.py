# deep learning libraries
import torch
from torch.utils.data import DataLoader
from torch.jit import RecursiveScriptModule

# other libraries
import pytest

# own modules
from src.utils import load_data
from src.train_functions import t_step

# set device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


@pytest.mark.order(3)
@pytest.mark.parametrize("data_path", ["data"])
def test_accuracy(data_path: str) -> None:
    """
    This is the test for the accuracy in the test set.
    """

    test_data: DataLoader
    _, _, test_data = load_data(data_path, batch_size=64)

    # define model
    model: RecursiveScriptModule = torch.jit.load("models/best_model.pt").to(device)

    # call evaluate
    accuracy_value: float = t_step(model, test_data, device)
    print(f"Accuracy: {accuracy_value}")

    # check if accuracy is higher than 92%
    assert accuracy_value > 0.92, "Accuracy not higher than 92%"

    # check if accuracy is higher than 95%
    assert accuracy_value > 0.95, "Accuracy not higher than 95%"

    # check if accuracy is higher than 92%
    assert accuracy_value > 0.97, "Accuracy not higher than 97%"

    return None
