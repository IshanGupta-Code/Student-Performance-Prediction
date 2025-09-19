
# Student Performance Prediction

A simple linear regression project to predict students' final exam scores based on the number of hours they study per week. This repository includes model training, evaluation, and visualization to understand how study hours impact performance.

---

## 🛠️ Features

- Linear Regression model built using **scikit-learn**  
- Evaluation metrics: MAE, MSE, RMSE, R²  
- Data visualization with Matplotlib  
  - Distribution of final exam scores (histogram)  
  - Regression line vs actual scores (scatter plot)  
- Prediction functionality for new study hour inputs  

---

## 📁 Repository Structure

```
Student-Performance-Prediction/
├── Data/
│   └── Students Performance Dataset.csv   # Dataset used for training & visualization
├── Project2.py                            # Core script: data load, model training, evaluation & plots
└── README.md                              # This file
```

---

## 📊 How It Works

1. **Load the Data**: Reads `Students Performance Dataset.csv` (features: `Study_Hours_per_Week`, target: `Final_Score`).  
2. **Train a Linear Regression Model**: Fits model to predict final score using study hours.  
3. **Evaluate the Model**:  
   - Mean Absolute Error (MAE)  
   - Mean Squared Error (MSE)  
   - Root Mean Squared Error (RMSE)  
   - R² Score (coefficient of determination)  
4. **Visualize Results**:  
   - Histogram of actual final scores  
   - Scatter plot of study hours vs actual scores with regression line  
5. **Make Predictions**: Predict final score for a given number of study hours (e.g. 9.5 hrs/week)  

---

## ⚙️ Usage

To run this project locally:

1. Clone the repo:

   ```bash
   git clone https://github.com/IshanGupta-Code/Student-Performance-Prediction.git
   cd Student-Performance-Prediction
   ```

2. Install required dependencies (if you don’t have them):

   ```bash
   pip install numpy pandas scikit-learn matplotlib
   ```

3. Run the main script:

   ```bash
   python Project2.py
   ```

4. (Optional) To predict a custom study hour value, edit the `new_hours` variable inside `Project2.py` and re-run.
