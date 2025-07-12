# Project 4

The objective of this fourth laboratory session is to implement the forward and backwrad of an RNN and an LSTM (with the nn package) to predict the day-ahead electricity price.

This project will be graded in the following way, 9 points will be graded by automatic tests, that can be verified by the criteria of a professor if it is necessary. The remaining 1 point will come from the inspection of the code by a professor. 

# Structure of the repo
In this repo, all the functional code is inside the src/ folder. For training the model and save it, the following module must be run:

    python -m src.train

In the src.train module there are some hyperparameters at the beginning of the main function that can be modified to optimize the model. For evaluating the model the following module can be run:

    python -m src.evaluate

The evaluate file will load a model called best_model.pt inside the models/ folder compute accuracy in the test set and print it in the command line. 


# Automatic tests (QA)
These tests are already provided to you, you can run them by executing:

    source test.sh

This way you will run the tests, the mypy checking and the flake8. The grades achieved by these tests can be overridden by the grade from a professor if the test is fulfilled but the goal of the function/class to be completed is not reached.

# Inspection of the code

This 1 point is meant to assess things that cannot be measured by automatic testing, such as code style and organization.


# Parts of the project

We strongly recommend the student follow the order stated here since some tests will depend on functions that you should have completed before.

### RNN forward (1.5 points)

This function is contained in the src.models module. You have to implement the forward of an RNN of 1 layer using a ReLU as a non-linearity. You cannot use the nn package here and only 1 for loop is allowed.

### RNN backward (3.5 points)

This function is contained in the src.models module. You have to implement the backward of an RNN of 1 layer using a ReLU as a non-linearity. You cannot use the nn package here and only 1 for loop is allowed.

### Dataset (1 point)

This class is contained in the src.data module.

### load_data function (1 point)

This function is contained in the src.data module. You should use the last 42 weeks of the training for the validation. Take into account that you should concatenate the end of the previous dataset to the next one for predicting the first value.

### Performance (2 points)

This is not a specific function but a performance you should achieve with your best model in the test set. A MAE lower than 2.7 would be 0.5 points, lower than 2.35 would be 1.5 points and lower than 2.2 would be 2 points. Now you can use the nn package and you should use a LSTM model.

### Mypy type checking (0.5 points)

The code must be properly typed hinted and pass the mypy type checker. This checker assures the fulfillment of coding standards in python:https://mypy.readthedocs.io/en/stable/

Following the coding guidelines is important since it allows the development of understandable and more robust source code. Static analysis of the source code allows the finding of potential issues such as:
- functions with too many input parameters
- infinite loops
- dead code
- variable with names that were repeated in the file
- etc.

To run the checker, the following command must be executed:

    mypy --cache-dir=/dev/null --check-untyped-defs --ignore-missing-imports .

It is also available in the test.sh file, that can be executed as follows:

    source test.sh

The benefit of this second file is that by running the pre_commit your code will also be formatted.

### Flake 8 test (0.5 points)

The code must follow the PEP-8 standard.

To run the checker the following command must be executed:

    flake8 --max-line-length 89

It is also available in the test.sh file, that can be executed as follows:

    source test.sh


