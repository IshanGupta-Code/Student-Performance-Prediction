from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Loading Data
data = pd.read_csv("Data/Students Performance Dataset.csv")

# Define Input and Output
X = data[['Study_Hours_per_Week']]
y = data['Final_Score']

# Train Model
model = LinearRegression()
model.fit(X,y)
pred_score = model.predict(X)

# Valid Regression Matrix
mae = mean_absolute_error(y, pred_score)
mse = mean_squared_error(y, pred_score)
rsme = np.sqrt(mse)
r2 = r2_score(y, pred_score)

print(
    "Mean Absolute Error(MAE):", round(mae, 2),
    "\nMean Squared Value(MSE):", round(mse, 2) ,
    "\nRoot Squared Mean Error(RSME):", round(rsme, 2),
    "\nR^2 Score(Model Accuracy):", round(r2, 4)  # Closer to 1 is better
)

# Histogram
plt.figure(figsize=(10,6))
plt.hist(data['Final_Score'], bins=30, color='skyblue', edgecolor='black')
plt.title('Distribution of FINAL EXAM SCORE')
plt.xlabel('Final Exam Score')
plt.ylabel('No. of Student')
# plt.grid(True)
plt.show() 

# Regression + Scatter Plot
plt.figure(figsize=(10,6))
plt.scatter(X, y, color='blue', label='Actual Score') #type:ignore
plt.plot(X, pred_score, color='red', label='Predicted Score (Regression Line)') #type:ignore
plt.title('Model Prediciton vs Actual Score')
plt.xlabel('Study Hours Per Week')
plt.ylabel('Final Output')
# plt.grid(True)
plt.show() 

new_hours = 9.5
new_pred_score = model.predict([[new_hours]])  #type:ignore
print(f"Predicted final score for {new_hours} =", new_pred_score)
