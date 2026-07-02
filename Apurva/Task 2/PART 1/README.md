# Linear Regression from Scratch using Gradient Descent

## Overview

The objective of this task was to understand how Linear Regression works internally by implementing the complete algorithm from scratch instead of using the built-in LinearRegression class from scikit-learn. The main focus was to learn how Gradient Descent updates the model parameters and gradually finds the best-fit line for a given dataset.

## Steps Performed

- Generated a synthetic dataset of 100 samples using NumPy.
- Visualized the generated data using a scatter plot.
- Split the dataset into training and testing sets.
- Created a custom LinearRegressionGD class to implement Linear Regression from scratch.
- Initialized the weight and bias to zero.
- Used Gradient Descent to update the weight and bias in every epoch by calculating their gradients.
- Calculated and stored the Mean Squared Error (loss) after each iteration to monitor the learning process.
- Predicted outputs for both the training and testing datasets.
- Visualized the fitted regression line along with the training data.
- Experimented with different learning rates (0.001, 0.01 and 0.1) and different numbers of epochs (500, 1000 and 2000).
- Compared the performance of each model using MAE, MSE, RMSE and R² score.
- Plotted the Loss vs Epochs graph to observe how the model converged during training.

## Algorithm Used

I implemented Linear Regression using Gradient Descent with the following approach:

1. Initialize the weight and bias to zero.
2. Predict the output using the equation:

   **y = wx + b**

3. Compute the Mean Squared Error (MSE) loss.
4. Calculate the gradients of the loss with respect to the weight and bias.
5. Update the parameters using the Gradient Descent update rule.
6. Repeat the process for the specified number of epochs until the loss decreases and the model converges.

This helped me understand how the model gradually improves its predictions by minimizing the error at each iteration.

## Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn (only for train-test split and evaluation metrics)

## Observations

The loss decreased consistently as the epochs increased, showing that Gradient Descent was working correctly. A learning rate of **0.001** converged slowly and required more epochs, while **0.01** and **0.1** reached good results much faster. The training performance was slightly better than the testing performance, which is expected because the model is trained on the training dataset. The small difference between the two results is mainly due to the random noise added while generating the synthetic data.

## Conclusion

Implementing Linear Regression from scratch gave me a much better understanding of how the algorithm works internally. Instead of relying on a pre-built library, I learned how the weight and bias are updated using Gradient Descent, how the loss reduces over time, and how different learning rates affect the speed of convergence. Overall, the model was able to learn the underlying linear relationship in the dataset and produced satisfactory results on both the training and testing data.