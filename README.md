# Machine Learning Recycling Project

### What is this project?

This project attempts to deploy a CNN on an edge computing device (Jetson Nano) such that the device can detect recycling objects and classify them into categories. Then they can be spilt up into seperate bins ready for recycling. 

This is important for different councils in the UK where different recycling rules exist. For example, in Wokingham glass cannot be recycled in the normal recycling bins but in Cambridge it can.

In addition to creating the CNN, this project also attempts to try out various optimisation methods on the CNN to increase the speed whilst keeping the accuracy the same.


### Current Progress

* A dataset has been collected from well known data sources.
* An initial model has been trained that achieves 90%+ accuracy
* This model has been integragted into a larger system such that an object can be detected and passed to this CNN from a video feed for classification
* Optimisation methods have been attempted. Including L1 Pruning, Taylor Pruning, APoZ Pruning and Teacher-Student Methods.
    * The code for these optimisation methods can be found in branches.
* An original dataset is in the process of being collected so that the edge system can be deployed and tested in a recycling bin

### Blogs

In addition to this readme, I have written multiple blogs on this project which can be found on https://www.nathanbaileyw.com/project-recycling-project

### Where is the code?

The majority of the code is located in the following files:

* main.py - main entry to train the CNN classification network
* network.py - CNN classification network
* dataset.py - Custom dataset class
* train_test.py - Functions to train and test the CNN classification network
* full_system_jetson.py - An example of the CNN classification network deployed in a system so that it can take a video input and classify video frames into classes
* get_council_data.py - A script to grab what can or cannot be recycled based on your postcode




