### Salary vs Years of Experience: Linear Regression from Scratch

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Load the dataset into a dataframe
data = pd.read_csv(r'C:\Users\DARA2\Downloads\experience_salary.csv')
#Check the columns in the dataset
print(data.columns)
X= data['years_experience'].values
y= data['salary'].values

#Initializing parameters
w= 0
b= 0
lr = 0.01  # learning rate
epochs = 4000 # number of iteration for learning
n= len(X)  # number of data points

## Training loop
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

print(len(loss_history))
print(loss_history[:5])
# Plot to check if Loss is decreasing per Epochs

plt.plot(loss_history)
plt.yscale('log')
plt.xlabel('Epochs')
plt.ylabel('Loss(log scale)')
plt.title('Loss vs Epochs')
plt.show()


import matplotlib.pyplot as plt

# Scatter plot (actual data)
plt.scatter(X, y, label="Actual Data")

# Line plot (model prediction)
plt.plot(X, y_pred, label="Fitted Line", color='red')

plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Linear Regression Fit")
plt.legend()
plt.show()

