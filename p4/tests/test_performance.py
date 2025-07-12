# deep learning libraries
import torch
from torch.utils.data import DataLoader
from torch.jit import RecursiveScriptModule

# other libraries
import pytest

# own modules
from src.data import load_data
from src.train_functions import t_step
from src.utils import set_seed

# set device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


@pytest.mark.order(5)
@pytest.mark.parametrize("data_path", ["data"])
def test_mae(data_path: str) -> None:
    """
    This is the test for the accuracy in the test set.
    """

    # set seed
    set_seed(42)

    test_data: DataLoader
    _, _, test_data, mean, std = load_data(data_path, batch_size=1)

    # define model
    model: RecursiveScriptModule = torch.jit.load("models/best_model.pt").to(device)

    # call evaluate
    mae_value: float = t_step(model, test_data, mean, std, device)

    # check if MAE is lower than 2.7
    assert mae_value < 2.7, "MAE is not lower than 2.7"

    # check if MAE is lower than 2.35
    assert mae_value < 2.35, "MAE is not lower than 2.35"

    # check if MAE is lower than 2.1
    assert mae_value < 2.2, "MAE is not lower than 2.2"

    return None
