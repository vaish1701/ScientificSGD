# Scientific Computation and Visualization (CSCI 596) - Parallelization of Stochastic Gradient Descent (SGD) Algorithm for Linear Regression.

The problem of data scarcity is very important since data are at the core of any AI project. The size of a dataset is often responsible for poor performances in ML projects, especially in case of Supervised machine learning. 

In this project, we have tackled the above issue via parallelization using MPI. We have observed how data and gradient update parallelization can help in scenarios where the training data is limited. 

SGD algorithm was parallelized using MPI library for linear regression to predict the Lyft/Uber cab price based on given data. Data was distributed across multiple processes and gradients were collected and calculated across processes. 

# Flowchart 

![](/flow_chart.jpeg)

# Results 
MSE without parallelization was 39.91. Post parallelization we were able to train the model efficiently and still were able to achieve the same MSE ranging between 39-40. 

Parameters used for training the model: Learning Rate = 0.01, Number of iterations = 1000, Batch Size = 500
 
![](/single.png)

Above image corresponds to executing SGD on a single process(without parallelization), showing the MSE value as 39.91. 
 
![](/screenshot1.png) 

Above is the stats for running SGD across 4 processes. Each section corresponds to the execution detail of a single process. The initial batch of size 500, has been divided across 4 processes with each process handling 125 data points. We were able to observe the data points alloted were distinct, confirming the data distribution(line 4 in each section). Loss attained across all the processes stays between 39-40 without any fluctuation, thereby achieving the goal. 

![](/Screenshot2.png) 

SGD was run across 2 processes and the above information was obtained. The image shows how the final gradient has been calculated by averaging its' own gradient value and the gradient values gathered from the neighbours. 

# Future Scope 
SGD forms the basis for most of the machine learning models and the implementation can be leveraged for any algorithm that makes use of SGD as optimizer. 

# Dataset
https://www.kaggle.com/brllrb/uber-and-lyft-dataset-boston-ma


