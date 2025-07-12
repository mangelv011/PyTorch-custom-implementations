# standard libraries
import os
import random
import requests
import tarfile
import shutil
from requests.models import Response
from tarfile import TarFile

# 3pps
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torch.jit import RecursiveScriptModule
from PIL import Image


class ImagenetteDataset(Dataset):
    """
    This class is the Imagenette Dataset.
    """

    def __init__(self, path: str) -> None:
        """
        Constructor of ImagenetteDataset.

        Args:
            path: Path of the dataset.

        Returns:
            None.
        """

        # TODO
        super().__init__()
       
        self.images = os.listdir(path)
        self.labels = [int(img.split("_")[0]) for img in self.images]
        self.images = [os.path.join(path, img) for img in self.images]
        


    def __len__(self) -> int:
        """
        This method returns the length of the dataset.

        Returns:
            Length of dataset.
        """

        # TODO
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        """
        This method loads an item based on the index.

        Args:
            index: index of the element in the dataset.

        Returns:
            Image tensor. Dimensions: [channels, height, width].
            Label of the image.
        """

        # TODO
        # el torch.tensor no funciona para imagenes PIL
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # valores estándar para ImageNet
                std=[0.229, 0.224, 0.225]
            )
        ])
        return transform(Image.open(self.images[index]).convert("RGB")), self.labels[index]


def load_imagenette_data(
    path: str, batch_size: int = 128, num_workers: int = 0
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    This function returns two Dataloaders, one for train data and
    other for validation data for imagenette dataset.

    Args:
        path: Path of the dataset.
        color_space: Color_space for loading the images.
        batch_size: Batch size for dataloaders. Default value: 128.and
        num_workers: Number of workers for loading data.
            Default value: 0.

    Returns:
        Train dataloader.
        Val dataloader.
        Test dataloader.
    """

    # download folders if they are not present
    if not os.path.isdir(f"{path}"):
        # create main dir
        os.makedirs(f"{path}")

        # download data
        download_data(path)

    # create datasets
    train_dataset: Dataset = ImagenetteDataset(f"{path}/train")
    val_dataset: Dataset
    train_dataset, val_dataset = random_split(train_dataset, [0.8, 0.2])
    test_dataset: Dataset = ImagenetteDataset(f"{path}/val")

    # define dataloaders
    train_dataloader: DataLoader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, persistent_workers=True,  prefetch_factor=2
    )
    val_dataloader: DataLoader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, persistent_workers=True,  prefetch_factor=2
    )
    test_dataloader: DataLoader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, persistent_workers=True,  prefetch_factor=2
    )

    return train_dataloader, val_dataloader, test_dataloader


def download_data(path: str) -> None:
    """
    This function downloads the data from internet.

    Args:
        path: Path to dave the data.
    """

    # define paths
    url: str = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz"
    target_path: str = f"{path}/imagenette2.tgz"

    # download tar file
    response: Response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(target_path, "wb") as f:
            f.write(response.raw.read())

    # extract tar file
    tar_file: TarFile = tarfile.open(target_path)
    tar_file.extractall(path)
    tar_file.close()

    # create final save directories
    os.makedirs(f"{path}/train")
    os.makedirs(f"{path}/val")

    # define resize transformation
    transform = transforms.Resize((224, 224))

    # loop for saving processed data
    list_splits: tuple[str, str] = ("train", "val")
    for i in range(len(list_splits)):
        list_class_dirs = sorted(os.listdir(f"{path}/imagenette2/{list_splits[i]}"))
        for j in range(len(list_class_dirs)):
            list_dirs = os.listdir(
                f"{path}/imagenette2/{list_splits[i]}/{list_class_dirs[j]}"
            )
            for k in range(len(list_dirs)):
                image = Image.open(
                    f"{path}/imagenette2/{list_splits[i]}/"
                    f"{list_class_dirs[j]}/{list_dirs[k]}"
                )
                image = transform(image)
                if image.im.bands == 3:
                    image.save(f"{path}/{list_splits[i]}/{j}_{k}.jpg")

    # delete other files
    os.remove(target_path)
    shutil.rmtree(f"{path}/imagenette2")

    return None


@torch.no_grad()
def parameters_to_double(model: torch.nn.Module) -> None:
    """
    This function transforms the model parameters to double.

    Args:
        model: Torch model.
    """

    # TODO
    for param in model.parameters():
        param.data = param.data.double()


class Accuracy:
    """
    This class is the accuracy object.

    Attributes:
        correct: Number of correct predictions.
        total: Number of total examples to classify.
    """

    correct: int
    total: int

    def __init__(self) -> None:
        """
        This is the constructor of Accuracy class. It should
        initialize correct and total to zero.
        """

        self.correct = 0
        self.total = 0

    def update(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        """
        This method update the value of correct and total counts.

        Args:
            logits: Outputs of the model.
                Dimensions: [batch, number of classes]
            labels: Labels of the examples. Dimensions: [batch].
        """

        # compute predictions
        predictions = logits.argmax(1).type_as(labels)

        # update counts
        self.correct += int(predictions.eq(labels).sum().item())
        self.total += labels.shape[0]

        return None

    def compute(self) -> float:
        """
        This method returns the accuracy value.

        Returns:
            accuracy value.
        """

        return self.correct / self.total

    def reset(self) -> None:
        """
        This method resets to zero the count of correct and total number of
        examples.
        """

        # init to zero the counts
        self.correct = 0
        self.total = 0

        return None


def save_model(model: torch.nn.Module, name: str) -> None:
    """
    This function saves a model in the 'models' folder as a torch.jit.
    It should create the 'models' if it doesn't already exist.

    Args:
        model: Torch model.
        name: Name of the model (without the extension, e.g. name.pt).
    """

    # create folder if it does not exist
    if not os.path.isdir("models"):
        os.makedirs("models")

    # save scripted model
    model_scripted: RecursiveScriptModule = torch.jit.script(model.cpu())
    model_scripted.save(f"models/{name}.pt")

    return None


def load_model(name: str) -> RecursiveScriptModule:
    """
    This function is to load a model from the 'models' folder.

    Args:
        name: Name of the model to load.

    Returns:
        Model in torchscript.
    """

    # define model
    model: RecursiveScriptModule = torch.jit.load(f"models/{name}.pt")

    return model


def set_seed(seed: int) -> None:
    """
    This function sets a seed and ensure a deterministic behavior.

    Args:
        seed: Seed number to fix radomness.
    """

    # set seed in numpy and random
    np.random.seed(seed)
    random.seed(seed)

    # set seed and deterministic algorithms for torch
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    # Ensure all operations are deterministic on GPU
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # for deterministic behavior on cuda >= 10.2
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    return None
