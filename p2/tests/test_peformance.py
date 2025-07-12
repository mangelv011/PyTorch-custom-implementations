# 3pps
import pytest

# own modules
from src.evaluate import main


@pytest.mark.order(8)
def test_accuracy() -> None:
    """
    This is the test for the accuracy in the test set.
    """

    # call evaluate
    accuracy_value: float = main("best_model")

    # check if accuracy is higher than 60%
    assert accuracy_value > 0.60, "Accuracy not higher than 60%"

    # check if accuracy is higher than 65%
    assert accuracy_value > 0.65, "Accuracy not higher than 65%"

    # check if accuracy is higher than 70%
    assert accuracy_value > 0.70, "Accuracy not higher than 70%"

    return None
