# deep learning libraries
import torch
from torch.utils.data import DataLoader

# other libraries
import pytest

# own modules
from src.data import download_data, ElectricDataset, load_data

# static variables
DATA_PATH: str = "data"


@pytest.mark.order(3)
def test_dataset() -> None:
    """
    This function is the test for ElectricDataset class.
    """

    # load dataset
    df_train, _ = download_data(path=DATA_PATH)
    dataset: ElectricDataset = ElectricDataset(df_train, 7)

    # check length
    assert (
        len(dataset) == 1449
    ), f"Incorrect length, expected 1449 and got {len(dataset)}"

    # check objects
    objects: tuple[torch.Tensor, torch.Tensor] = dataset[0]

    # check number of elements
    assert (
        len(objects) == 2
    ), f"Incorrect number of elements for the output, expected 2 got {len(objects)}"

    # check past values dimension
    assert objects[0].shape == (
        7,
        24,
    ), (
        f"Incorrect shape for the past values, expected (7, 24) and got "
        f"{objects[0].shape}"
    )

    # check past values
    assert (
        objects[0]
        != torch.from_numpy(df_train["Price"].to_numpy()[:7 * 24]).view(-1, 24)
    ).sum().item() == 0, "Incorrect past values"

    # check currect values dimension
    assert objects[1].shape == (
        24,
    ), f"Incorrect shape for the past values, expected (24,) and got {objects[1].shape}"

    # check current values
    assert (
        objects[1]
        != torch.from_numpy(df_train["Price"].to_numpy()[7 * 24 : 7 * 24 + 24])
    ).sum().item() == 0, "Incorrect current values"

    return None


@pytest.mark.order(4)
def test_load_data() -> None:
    """
    This function is the test for load_data function.
    """

    # load data
    train_data: DataLoader
    val_data: DataLoader
    test_data: DataLoader
    train_data, val_data, test_data, _, _ = load_data(DATA_PATH, batch_size=1)

    # check size of train_data
    assert (
        len(train_data) == 1155
    ), f"Incorrect length of training set, expected 1155 and got {len(train_data)}"

    # check size of val_data
    assert (
        len(val_data) == 294
    ), f"Incorrect length of training set, expected 294 and got {len(val_data)}"

    # check size of test_data
    assert (
        len(test_data) == 728
    ), f"Incorrect length of training set, expected 728 and got {len(test_data)}"

    # check mean
    assert (
        torch.concat(
            (train_data.dataset.dataset, val_data.dataset.dataset[7:]),  # type: ignore
            dim=0,
        )
        .mean()
        .round(decimals=4)
        .item()
        == 0
    ), "Incorrect mean"

    # check std
    assert (
        torch.concat(
            (train_data.dataset.dataset, val_data.dataset.dataset[7:]),  # type: ignore
            dim=0,
        )
        .std()
        .round(decimals=4)
        .item()
        == 1
    ), "Incorrect std"

    return None
