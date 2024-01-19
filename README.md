# Machine Learning Recycling Project

### What is this branch?

This branch implements Taylor pruning of the classification network implemented in the main branch. For more inforamtion on this pruning method can be found here: https://arxiv.org/abs/1611.06440, https://jacobgil.github.io/deeplearning/pruning-deep-learning

I also wrote a blog on pruning techniques which can be found here: https://nathanbaileyw.medium.com/machine-learning-recycling-project-part-2-research-383602941b70

### Where is the code?

The majority of the code is located in the following files:

* main_prune_taylor.py - Main entry file that contains all the logic for pruning
* network.py - CNN classification network
* dataset_taylor_data.py - (modified to compute metrics for Taylor Pruning)
* dataset.py - Custom dataset class
* train_test.py - Functions to train and test the CNN classification network
