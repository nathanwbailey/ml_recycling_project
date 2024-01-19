# Machine Learning Recycling Project

### What is this branch?

This branch implements Teacher-Student methods with the classification network implemented in the main branch. More information on this optimisation method can be found here: https://arxiv.org/abs/1312.6184, https://arxiv.org/abs/1503.02531

I also wrote a blog on this method which can be found here: https://nathanbaileyw.medium.com/machine-learning-recycling-project-part-2-research-383602941b70

### Where is the code?

The majority of the code is located in the following files:

* main_student_teacher.py - Main entry file that contains all the logic for this optimisation method
* network_teacher_student - CNN classification network (modified to include the teacher-student logic)
* dataset_teacher_student.py - Custom dataset class that includes the labels for teacher-student method
* train_test_teacher_student.py - Functions to train and test the CNN classification network (modified to include the teacher-student logic)

APoZ Pruning was also done with this method. Code files for that can be found here:

* main_prune_apoz.py - Main entry file that contains all the logic for this optimisation method + APoZ Pruning
* network_apoz_teacher_student - CNN classification network (modified to include the teacher-student logic and APoZ pruning)
* dataset_apoz.py - (modified to compute metrics for APoZ Pruning)
* train_test.py - Functions to train and test the CNN classification network
