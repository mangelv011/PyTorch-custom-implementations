# deep learning libraries
import torch
import torchvision
import numpy as np
import pandas as pd
from torch.jit import RecursiveScriptModule
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

# other libraries
import os
import random
import torch.nn.functional as F




def set_seed(seed: int) -> None:
    """
    This function sets a seed and ensure a deterministic behavior

    Args:
        seed: seed number to fix radomness
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





def get_dropout_random_indexes(shape, p: float) -> torch.Tensor:
    """
    This function get the indexes to put elements at zero for the
    dropout layer. It ensures the elements are selected following the
    same implementation than the pytorch layer.

    Args:
        shape: shape of the inputs to put it at zero. Dimensions: [*].
        p: probability of the dropout.

    Returns:
        indexes to put elements at zero in dropout layer.
            Dimensions: shape.
    """

    # get inputs indexes
    inputs: torch.Tensor = torch.ones(shape)

    # get indexes
    indexes: torch.Tensor = F.dropout(inputs, p)
    indexes = (indexes == 0).int()

    return indexes