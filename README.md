# Machine Learning Recycling Project

### What is this branch?

This branch implements APoZ pruning of the classification network implemented in the main branch. For more inforamtion on this pruning method can be found here: https://arxiv.org/abs/1607.03250

I also wrote a blog on pruning techniques which can be found here: https://nathanbaileyw.medium.com/machine-learning-recycling-project-part-2-research-383602941b70

### Where is the code?

The majority of the code is located in the following files:

* main_prune_apoz.py - Main entry file that contains all the logic for pruning
* apoz_prune.py - Contains functions for calculating the APoZ metrics
* network_apoz.py - CNN classification network (modified to be compatible with 
APoZ)
* dataset_apoz.py - (modified to compute APoZ metrics)
* dataset.py - Custom dataset class
* train_test.py - Functions to train and test the CNN classification network
