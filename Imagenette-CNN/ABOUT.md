# Project 2

The objective of this second laboratory session is to implement a CNN (convolutional neural network) to be trained and evaluated over the [Imagenette](https://github.com/fastai/imagenette) dataset (which is a small subset of the known [ImageNet](https://www.image-net.org/) dataset). You will have to implement the required functions to pass the proposed tests.

This project will be graded in the following way, 8 points will be graded by automatic tests, that can be verified by the criteria of a professor if it is necessary. The remaining 2 points will come from the inspection of the code by a professor.

If you want more information about the mathematical explanation for gradients you can look into this [link]( https://web.eecs.umich.edu/~justincj/teaching/eecs442/notes/linear-backprop.html).

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

These 3 points are meant to assess things that cannot be measured by automatic testing, such as code style and organization.


# Parts of the project

We strongly recommend the student follow the order stated here since some tests will depend on functions that you should have completed before.

### Imagenette dataset class (0.5 points)

This class is contained in the src.utils module. This class is the dataset to load images and labels from the memory and then, combined with the dataloader, creates the batches to train the model.

### parameters_to_double function (0.5 points)

This function is contained in the src.utils module. This function changes the dtype of the parameters of a model, from float to double. It will be needed for the test of the conv layer, since otherwise the output of this layer will not be equal to the default implementation of pytorch due to underflow errors.

### ReLUFunction class (0.5 points)

This class is contained in the src.models module. This class implements the forward and backward of the ReLU layer. To implement it you cannot use the nn pytorch module or pytorch custom operations (use only slicing and indexing).

### LinearFunction class (1 point)

This class is contained in the src.models module. This class implements the forward and backward of the Linear layer. To implement it you cannot use the nn pytorch module or pytorch custom operations (use only torch.matmul, slicing and indexing).

### Conv2dFunction class (2.5 points)

This class is contained in the src.models module. This class implements the forward and backward of the Conv2d layer. To implement it you cannot use the nn pytorch module, except fold and unfold functions.

### Block class (0.4 points)

This class is contained in the src.models module. You can from now on use the nn pytorch module. Use a Sequential to encapsulate all the layers.

### CNNModel class (0.1 points)

This is the model you will use to classify. It is recommended to use the Block class constructed before.

### Main function in main and evaluate (0.5 points)

These are the main functions of src.train module and src.evaluate. The evaluate one should be tested. It is strongly recommended to use the Accuracy class from the src.utils module for both modules and plot the accuracy and loss of the train and eval in the training.


### Performance (1 points)
This is not a specific function but a performance you should achieve with your best model in the test set. A performance higher than 60% would have a score of 0.2 points, higher than 65% 0.5 and higher than 70% 1 point. You should try to play with the hyperparameters and then rename your best model as best_model.pt inside the models' folder. Please note that the test implemented for the performance part will fail if your train model does not reach the accuracy thresholds defined above.


### Mypy type checking (1 points)

The code must be properly typed hinted and pass the mypy type checker. This checker assures the fulfillment of coding standards in python:https://mypy.readthedocs.io/en/stable/

Following the coding guidelines is important since it allows the development of understandable and more robust source code. Static analysis of the source code allows the finding of potential issues such as:
- functions with too many input parameters
- infinite loops
- dead code
- variable with names that were repeated in the file
- etc.

To run the checker, the following command must be executed:

    mypy --cache-dir=/dev/null --check-untyped-defs --ignore-missing-imports .

It is also available in the pre_commit.sh file, that can be executed as follows:

    source test.sh

The benefit of this second file is that by running the pre_commit your code will also be formatted.

