# standard libraries
import os

# 3pps
import torch
import pytest

# own modules
from src.utils import download_data, ImagenetteDataset, parameters_to_double


@pytest.mark.order(1)
@pytest.mark.parametrize("path", ["data"])
def test_imagenette(path: str) -> None:
    """
    This fucntion is the test for Imagenette class.

    Args:
        path: path for saving the data
    """

    # download folders if they are not present
    if not os.path.isdir(f"{path}"):
        # create main dir
        os.makedirs(f"{path}")

        # download data
        download_data(path)

    # create datasets
    train_dataset: ImagenetteDataset = ImagenetteDataset(f"{path}/train")
    test_dataset: ImagenetteDataset = ImagenetteDataset(f"{path}/val")

    # check train length
    assert (
        len(train_dataset) == 9296
    ), f"Incorrect length, got {len(train_dataset)} and it should be 9296"

    # check test length
    assert (
        len(test_dataset) == 3856
    ), f"Incorrect length, got {len(test_dataset)} and it should be 3856"

    # get example of output
    element: tuple[torch.Tensor, int] = train_dataset[0]

    # check number of objects returned by __getitem__
    assert len(element) == 2, (
        f"Incorrect number of objects returned by __getitem__ method, "
        f"2 were expected and got {len(element)}"
    )

    # check first object type
    assert isinstance(
        element[0], torch.Tensor
    ), "Incorrect object type of first element of __getitem__ output"

    # check first object type
    assert isinstance(
        element[1], int
    ), "Incorrect object type of second element of __getitem__ output"

    # check first object
    assert element[0].shape == (3, 224, 224), "Incorrect shape of image tensor"

    return None


@pytest.mark.order(2)
def test_parameters_to_double() -> None:
    """
    This function is the test for the parameters_to_float function.

    Args:
        path: path for saving the data
    """

    # define model
    model: torch.nn.Module = torch.nn.Linear(32, 10)

    # pass parameters to double
    parameters_to_double(model)

    # iterate over parameters and check dtype
    for parameter in model.parameters():
        if parameter.data.dtype == torch.float32:
            assert (
                parameter.data.dtype != torch.float32
            ), f"Incorrect dtype, expected double found {parameter.data.dtype}"

    # define model
    model = torch.nn.Sequential(
        torch.nn.Conv1d(3, 40, 3),
        torch.nn.Linear(40, 20),
        torch.nn.ReLU(),
        torch.nn.Sequential(
            torch.nn.Linear(20, 10), torch.nn.ReLU(), torch.nn.Linear(10, 5)
        ),
    )

    # pass parameters to double
    parameters_to_double(model)

    # iterate over parameters and check dtype
    for parameter in model.parameters():
        if parameter.data.dtype == torch.float32:
            assert (
                parameter.data.dtype != torch.float32
            ), f"Incorrect dtype, expected double found {parameter.data.dtype}"

    return None
