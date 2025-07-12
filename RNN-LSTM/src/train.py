# deep learning libraries
import torch
from torch.utils.data import DataLoader  # noqa: F401
from torch.utils.tensorboard import SummaryWriter

# other libraries
from tqdm.auto import tqdm  # noqa: F401
from typing import Final

# own modules
from src.data import load_data
from src.models import MyModel
from src.train_functions import train_step, val_step
from src.utils import set_seed, save_model, parameters_to_double  # noqa: F401

# static variables
DATA_PATH: Final[str] = "data"

# set device and seed
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
set_seed(42)


def main() -> None:
    """
    This function is the main program for training.
    """
    # Hyperparameters
    hidden_size = 64
    past_days = 7
    batch_size = 128
    learning_rate = 0.001
    epochs = 50

    # Load data
    print("Loading data...")
    train_data, val_data, test_data, mean, std = load_data(
        save_path=DATA_PATH, past_days=past_days, batch_size=batch_size, shuffle=True
    )

    # Create model and move to device first
    print(f"Creating model with hidden size {hidden_size}...")
    model = MyModel(hidden_size=hidden_size).to(device)

    # Convert model to double precision
    model = model.double()

    # Create a new Loss on the same device
    loss_fn = torch.nn.L1Loss().to(device)

    # Create optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=1e-5
    )

    # Create scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.05, patience=40
    )

    # Set up tensorboard writer
    writer = SummaryWriter(log_dir="runs")

    # Training loop
    print(f"Starting training for {epochs} epochs...")
    _ = float("inf")  # Renamed best_loss to avoid unused variable warning

    # Process epochs
    for epoch in range(1, epochs + 1):
        # Training step
        train_step(
            model=model,
            train_data=train_data,
            mean=mean,
            std=std,
            loss=loss_fn,
            optimizer=optimizer,
            writer=writer,
            epoch=epoch,
            device=device,
        )

        # Validation step
        val_step(
            model=model,
            val_data=val_data,
            mean=mean,
            std=std,
            loss=loss_fn,
            scheduler=scheduler,
            writer=writer,
            epoch=epoch,
            device=device,
        )

    # Take model off device before saving
    model.cpu()
    save_model(model, "best_model")

    # Close tensorboard writer
    writer.close()

    print("Done!")


if __name__ == "__main__":
    main()
