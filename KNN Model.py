# Import all Libraries 
import pandas as pd
import numpy as  np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

# Reading Data From the Dataset

data_file = r'E:\Python\Libraries\Scikit-Learn\Data\titanic.csv'
data = pd.read_csv(data_file, encoding='latin1')
data.info()
print(data.isnull().sum())

# Data Cleaning and Feature Engineering
def preprocess_data(df):
    df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], inplace = True)

    df['Embarked'].fillna('S', inplace = True)
    df.drop(columns = ['Embarked'], inplace = True)

    fill_missing_age(df)

    # Convert Gender
    df['Sex'] = df['Sex'].map({'male':0 , 'female':1})

    # Feature Enginnering
    df["FamilySize"] = df['SibSp'] + df['Parch']
    df['IsAlone'] = np.where(df['FamilySize'] == 0 , 1, 0)
    df['FareBin'] = pd.qcut(df['Fare'], 4, labels=False)
    df['AgeBin'] = pd.cut(df['Age'], bins=[0,12,20,40,60, np.inf], labels=False)
    df['Fare'].fillna(df['Fare'].median(), inplace=True)

    return df

# Fill in Missing Ages
def fill_missing_age(df):
    age_fill_map = {}
    for pclass in df['Pclass'].unique():
        if pclass not in age_fill_map:
            age_fill_map[pclass] = df[df['Pclass'] == pclass]['Age'].median()
            
    df['Age'] = df.apply(lambda row: age_fill_map[row['Pclass']] if pd.isnull(row['Age']) else row['Age'], axis = 1)

data = preprocess_data(data)
print("NaNs after preprocessing:\n", data.isnull().sum())
data = data.fillna(0)  # or use another strategy

# Create Feature / Target Variables (Make Flashcards)
X = data.drop(columns = ['Survived'])
y = data['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# ML Preprocessing
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# HyperParameter Tuning - KNN (K Nearest Neighbours)
def tune_model(X_train, y_train):
    param_grid = {
        'n_neighbors': range(1, 21),
        'metric': ['euclidean', 'manhattan', 'minkowski'],
        'weights': ['uniform', 'distance']
    }

    model = KNeighborsClassifier()
    grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_


best_model = tune_model(X_train, y_train)

# Prediction and Evaluation
def evaluate_model(model, X_test, y_test):
    prediction = model.predict(X_test)
    accuracy = accuracy_score(y_test, prediction)
    matrix = confusion_matrix(y_test, prediction)
    return accuracy, matrix

accuracy, matrix = evaluate_model(best_model, X_test, y_test)

print(f'Accuracy: {accuracy*100:.2f}%')
print(f'Confusion Matrix:')
print(matrix)

# Plot
def plot_model(matrix):
    plt.figure(figsize=(10,17))
    sns.heatmap(matrix, annot=True, fmt='d', xticklabels=['Survived', 'Not Survived'], yticklabels=['Not Survived', 'Survived'])
    plt.title("Confusion Matrix")
    plt.xlabel('Predicted Value')
    plt.ylabel('True Value')
    plt.show()

plot_model(matrix)