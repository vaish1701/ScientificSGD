# Scientific Computation and Visualization (CSCI 596)
# Parallelisation of Stochastic Gradient Descent (SGD) Algorithm for Linear Regression.

Parallelized SGD Algorithm implmentation using MPI Library for Linear Regression to predict the Lyft/Uber cab price based on given data 

Summary:
To observe how gradient update parallelisation can help in scenarios where the training data is limited.
Achieved this by distributing given data across multiple processes and culminating gradients calculated across processes.

Dataset: 
https://www.kaggle.com/brllrb/uber-and-lyft-dataset-boston-ma

MPI Methods:
Barrier, Scatter, Gather, Broadcast


