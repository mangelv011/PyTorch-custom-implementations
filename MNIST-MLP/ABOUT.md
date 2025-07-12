# Project 1

This project will be graded in the following way, 7 points will be graded by automatic tests, that can be verified by the criteria of a professor if it is necessary. The remaining 3 points will come from the inspection of the code by a professor.

The goal of this first lab is to implement the required source code to train and test a neural network to recognize ciphers based on the common dataset denominated MNIST (https://www.kaggle.com/datasets/hojjatk/mnist-dataset).

This dataset consists on a large set of handwritten digits. The provided source code is a pytorch project with the function to load data, train the data, test it and extract metrics. Parts of the source code is missing and you will have to complete it to be able to successfully train the network.

As commented below, we recommend you to implement the source code sequentially step by step and its corresponding test (QA or quality assurance) to evidence the correct behavior of each part that you develop. 

The first thing you will have to do is install the required dependencies, defined in the file "requirements.txt".

To log the metrics during training and validation, we will be using tensorboard. Install also the extension of vs-code for tensorboard. 

IMPORTANT: To submit it is only needed to push in github.

# Structure of the repo
In this repo, all the functional code is inside the src/ folder. For training the model and save it, the following module must be run:

    python -m src.train

In the src.train module there are some hyperparameters at the beginning of the main function that can be modified to optimize the model. For evaluating the model the following module can be run:

    python -m src.evaluate

The evaluate file will load a model called best_model.pt inside the models/ folder compute accuracy in the test set and print it in the command line. This is the only model that will be uploaded to github.


# Automatic tests (QA)
These tests are already provided to you, you can run them by executing:

    pytest .
    
At the beginning, these tests will fail and you will have to implement the required functions properly for them to be set as PASS. The grades achieved by these tests can be overridden by the grade from a professor if the test is fulfilled but the goal of the function/class to be completed is not reached.

# Inspection of the code

These 3 points are meant to assess things that cannot be measured by automatic testing, such as code style and organization.

# Parts of the project

We recommend the student follow the order we present in this section since it is the easiest and most natural one to complete the project.

### load_data function (QA test, 1 point)

This function is contained in the src.utils module and must be completed to load the three dataloaders of train, val and test in their respective order. In order to do that, the division between train and val must be 0.8-0.2. Finally, all batches should be equal in size.  

To execute the unit test implemented to ensure the correct implementation of the load_data function, run:
pytest tests/test_utils.py::test_load_data

This will be analogous to the rest of the functions.

### ReLU class (QA test, 1 point)

This class is contained in the src.models and must be completed using only matrix operations (slicing and indexing, operations as torch.where or torch.max are not allowed). The functionality must be the same as the torch.nn.ReLU class from PyTorch.

### Linear class (QA test, 1 point)

This class is contained in the src.models and must be completed using only matrix operations (slicing and indexing, operations as torch.where or torch.max are not allowed). The conventions used must be the PyTorch ones.

### MyModel class (QA test, 1 point)

This class is contained in the src.models and must be completed by calling the Linear and ReLU classes and PyTorch operations (without PyTorch NN layers).

### accuracy function (1 point)

This function is contained in the src.utils module and must be completed to compute the accuracy of the datasets.

### Train step (Inspection)

Implement the train functions to make the file train.py executable. This is the training step for each epoch. It should train the parameters, compute the average loss and accuracy and log them into tensorboard. Check the graphic shown in tensorboard for the train loss and make sure that it converges to zero after training has finished. This function is contained in the src.train_functions module.

### Validation step (Inspection)

This is the validation step for each epoch. It should compute the average loss and accuracy and log them into tensorboard. This function is contained in the src.train_functions module.

### Testing step (Inspection)

This is the test step for each epoch. It should compute the average accuracy and return it. This function is contained in the src.train_functions module.

### Performance (2 points)

This is not a specific function but a performance you should achieve with your best model in the test set. A performance higher than 94% would have a score of 1 point, higher than 96% 1.5 and higher than 97% 2 points. You should try to play with the hyperparameters and then rename your best model as best_model.pt inside the models' folder (this should be the only model uploaded to github). Please note that the test implemented for the performance part will fail if your train model does not reach the accuracy thresholds defined above.