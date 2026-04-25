### Salary vs Years of Experience: Linear Regression from Scratch

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

##LOADING THE DOCUMENT INTO A DATAFRAME
data = pd.read_csv(r'C:\Users\DARA2\Downloads\experience_salary.csv')
#Check the columns in the dataset
print(data.columns)
X= data['years_experience'].values
y= data['salary'].values

#INITIALIZING PARAMETERS, HYPERPARAMETERS
w= 0 # slope, measures the importance of a feature
b= 0 # intercept, determines where the line of best fit crosses the vertical axiswhen all input features are zero
lr = 0.01  # learning rate
epochs = 4000 # number of iteration for learning
n= len(X)  # number of data points

## TRAINING LOOP
# Model Hypothesis: I assume a linear relationship between the years of experience and salary
loss_history = []

for i in range(epochs):
    y_pred = w*X + b  # y: salary, w: slope(how much salary increases by years of experience), x: years of experience, b: intercept(starting salary)
    # Computing gradients to minimize loss function: error in predicted y and actual y (y-y_pred)
    dw = (-2/n) * np.sum(X*(y-y_pred)) # Computing partial derivative for the cost function
    db = (-2/n) * np.sum(y-y_pred)
    # Parameter Update; lr= step size, dw = direction + magnitude
    w -= lr * dw # How big a step you take to look for the best value at a global minimum
    b -= lr * db
    loss = np.mean((y-y_pred)**2)
    loss_history.append(loss)
    # Loss monitoring
    if i % 100 == 0:
        print(f'Epoch {i}, Loss {loss}')
        
print('Learned parameters')
print('w:', w)
print('b:', b)

##PLOTS
# Plot to check if Loss is decreasing per Epochs
plt.plot(loss_history)
plt.yscale('log')
plt.xlabel('Epochs')
plt.ylabel('Loss(log scale)')
plt.title('Loss vs Epochs')
plt.show()
plt.savefig('plots/loss_vs_epochs.png')

# Scatter plot (actual data) -- CHECKING REGRESSION FIT
plt.scatter(X, y, label="Actual Data")
# Line plot (model prediction)
plt.plot(X, y_pred, label="Fitted Line", color='red')

plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Linear Regression Fit")
plt.legend()
plt.show()
plt.savefig('plots/linear_fit.png')


## Using Prebuilt Linear Regression on Scikit-Learn
import sklearn.linear_model

model = sklearn.linear_model.LinearRegression()
model.fit(X.reshape(-1, 1), y)
print('Scikit-Learn Model: Learned parameters')
print('w:', model.coef_[0])
print('b:', model.intercept_)

# Scatter plot (actual data)
plt.scatter(X, y, label="Actual Data")
# Line plot (model prediction)
y_pred_sklearn = model.predict(X.reshape(-1, 1))
plt.plot(X, y_pred_sklearn, label="Fitted Line (Scikit-Learn)", color='green')
plt.xlabel("years_experience")
plt.ylabel("salary")
plt.title("Linear Regression Fit (Scikit-Learn)")
plt.legend()
plt.show()
plt.savefig('plots/linear_fit_sklearn.png')


## Evaluation Metrics
# R² (Coefficient of Determination): Measures the proportion of variance in the dependent variable that is predictable from the independent variable(s). R² values range from 0 to 1, where a value closer to 1 indicates a better fit of the model to the data.
from sklearn.metrics import r2_score

# Custom model
r2_custom = r2_score(y, y_pred)

# Sklearn model
r2_sklearn = r2_score(y, y_pred_sklearn)

print("Custom R²:", r2_custom)
print("Sklearn R²:", r2_sklearn)

# Plot Resiudals to check for patterns in the errors of the model
import matplotlib.pyplot as plt

residuals = y - y_pred

plt.scatter(X, residuals)
plt.axhline(y=0)
plt.xlabel("Years of Experience")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.show()
plt.savefig('plots/residual_plot.png')

plt.hist(residuals, bins=20)
plt.title("Residual Distribution")
plt.xlabel("Error")
plt.ylabel("Frequency")
plt.show()
plt.savefig('plots/residual_histogram.png')


## Final Model Visualization
plt.scatter(X, y, label="Actual Data")
plt.plot(X, y_pred, label="Custom Model")
plt.plot(X, y_pred_sklearn, linestyle='dashed', label="Sklearn Model")

plt.legend()
plt.title("Final Model Fit")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()
plt.savefig('plots/final_model_fit.png')
