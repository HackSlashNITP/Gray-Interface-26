# IMPLEMENTED LINEAR REGRESSION USING GRADIENT DESCENT IN PYTHON

## Aim of this task :
To understand how a regression model works internally (the mathematical mechanics behind linear regression).

## Workflow:
I simply followed four steps as mentioned below:

###### Step-1: Data generation 
Here, i generated a synthetic dataset taking the same example from task pdf we were assigned. And then further did shuffling in the dataset using the permutation function of numpy and splitted it into  seperate training and testing dataset using numpy slicing operations only because we were not allowed to use sklearn.

#### Step-2: Gradient Descent loop
Implemented a full parameterized training loop utilizing NumPy & Python. Then for each epoch in range of epochs calculated y_pred, the cost function, gradients of weights and bias then further updated these parameters.

#### Step-3 : Experimentation & Visualization
Generated a scatter plot of the actual synthetic data points and overlaid it with a line plot of the final optimized model predictions to verify the line of best fit visually.

Wrapped the training loop into a reusable Python function to conduct automated hyperparameter testing. then further evaluated multiple learning rates (i.e, for alpha = 0.1, 0.01, 0.001, 0.0001) across constant epochs to compare behavior.

Utilized Matplotlib to plot Loss (Cost) vs. Epochs for each run. This visually demonstrated model convergence and helped identify when a learning rate was too high or too low.

#### Evaluation Metrics:
Evaluated model generalizability on both training and testing subsets using raw mathematical implementations of four primary metrics:

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R-Squared Score ($R^2$)


## Key Observation:
Choosing an optimal learning rate (alpha = 0.01) allowed the model to converge smoothly. Setting alpha too low resulted in incredibly slow training, while a high learning rate caused unstable updates. 
