# Project 3

The objective of this third laboratory session is to implement a CNN (convolutional neural network) to be trained and evaluated over the [Imagenette](https://github.com/fastai/imagenette) dataset (which is a small subset of the known [ImageNet](https://www.image-net.org/) dataset), but now using every fetaure for optimzation and regularization. You will have to implement the required functions to pass the proposed tests.

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

This way you will run the tests and the mypy checking. The grades achieved by these tests can be overridden by the grade from a professor if the test is fulfilled but the goal of the function/class to be completed is not reached.

# Inspection of the code

This 1 point is meant to assess things that cannot be measured by automatic testing, such as code style and organization.


# Parts of the project

We strongly recommend the student follow the order stated here since some tests will depend on functions that you should have completed before.

### SGD class (1 point)

This class is contained in the src.optimization module. This class is a custom implementation of SGD algorithm. The recommendation is to look at the documentation: [link](https://pytorch.org/docs/stable/generated/torch.optim.SGD.html). To implement it you are not allowed to use pytorch methods (e.g. add, mul, div, etc).

### SGDMomentum class (1 point)

This class is contained in the src.optimization module. This class is a custom implementation of SGD algorithm with momentum. The recommendation is to look at the documentation: [link](https://pytorch.org/docs/stable/generated/torch.optim.SGD.html). To implement it you are not allowed to use pytorch methods (e.g. add, mul, div, etc).

### SGDNesterov class (1 point)

This class is contained in the src.optimization module. This class is a custom implementation of SGD algorithm with Nesterov momentum. The recommendation is to look at the documentation: [link](https://pytorch.org/docs/stable/generated/torch.optim.SGD.html). To implement it you are not allowed to use pytorch methods (e.g. add, mul, div, etc).
### Adam class (1 point)

This class is contained in the src.optimization module. This class is a custom implementation of Adam algorithm. The recommendation is to look at the documentation: [link](https://pytorch.org/docs/stable/generated/torch.optim.Adam.html). To implement it you are not allowed to use pytorch methods (e.g. add, mul, div, etc).

### StepLR scheduler class (1 point)

This class is contained in the src.utils module. This is a custom implementation of the StepLR sheduler from pytorch.

### Dropout class (1 point)

This class is contained in the src.models module. This is a custom implementation of the Dropout. 


### Performance (2 points)

This is not a specific function but a performance you should achieve with your best model in the test set. A performance higher than 65% would have a score of 0.5 point, higher than 70% 1.5 and higher than 75% 2 points. You should try to play with the hyperparameters and then rename your best model as best_model.pt inside the models' folder. Please note that the test implemented for the performance part will fail if your train model does not reach the accuracy thresholds defined above.


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


