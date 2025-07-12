"""
This module contains utils for testing functions.
"""

# Standard libraries
from typing import Any


def add_seed(parameters: tuple[Any, ...], num_seeds: int = 3) -> list[tuple[Any, ...]]:
    """
    This function adds the seed to a set of parameters to have a 
    set of combination.

    Args:
        parameters: Tuple of any parameters.
        num_seeds: Number of seed to add. Defaults to 3.

    Returns:
        List of parameters tuple.
    """

    return [(*parameters, seed) for seed in range(num_seeds)]
