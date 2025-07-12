"""
This module contains the code to evaluate the accuracy.
"""

# 3pps
import pytest

# Own modules
from src.evaluate import main


@pytest.mark.order(7)
def test_accuracy() -> None:
    """
    This is the test for the accuracy in the test set.
    """

    # Call evaluate
    accuracy_value: float = main("best_model")

    # Check if accuracy is higher than 65%
    assert accuracy_value > 0.65, "Accuracy not higher than 65%"

    # Check if accuracy is higher than 70%
    assert accuracy_value > 0.70, "Accuracy not higher than 70%"

    # Check if accuracy is higher than 75%
    assert accuracy_value > 0.75, "Accuracy not higher than 75%"

    return None
