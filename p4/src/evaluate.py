# deep learning libraries
import torch
from torch.utils.data import DataLoader  # noqa: F401
from torch.jit import RecursiveScriptModule  # noqa: F401

# other libraries
from typing import Final

# own modules
from src.data import load_data
from src.utils import set_seed, load_model
from src.train_functions import t_step

# static variables
DATA_PATH: Final[str] = "data"
NUM_CLASSES: Final[int] = 10

# set device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
set_seed(42)


def main() -> None:
    """
    This function is the main program.
    """
    # Using the same parameters as in train.py
    past_days = 7
    batch_size = 1
    model_name = "best_model"

    # Load data
    print(f"Loading data with {past_days} days of context...")
    _, _, test_data, mean, std = load_data(
        save_path=DATA_PATH,
        past_days=past_days,
        batch_size=batch_size,
        shuffle=False,  # No need to shuffle for testing
        drop_last=False,  # Use all data for testing
    )

    # Load model
    print(f"Loading model: {model_name}...")
    model = load_model(model_name)
    model.to(device)

    # Print model information
    print("Model loaded successfully.")
    print(f"Device: {device}")

    # Evaluate model using t_step
    print("Evaluating model on test data...")
    mae = t_step(model=model, test_data=test_data, mean=mean, std=std, device=device)

    print(f"Test Mean Absolute Error (MAE): {mae:.6f}")


if __name__ == "__main__":
    main()
