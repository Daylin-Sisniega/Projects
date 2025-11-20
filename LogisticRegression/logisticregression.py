# -*- coding: utf-8 -*-
"""
Created on Wed Nov  5 11:09:11 2025

@author: dayli
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report



#GET DATA
titanic_daylin = pd.read_csv(r"C:\Users\dayli\OneDrive\Escritorio\semester3\introductionai\LogisticRegression\titanic.csv")


#INITIAL EXPLORATION
#print it to see if it works (everything)
print("\n--- Records ---")
print(titanic_daylin.head())

#PRINT THE FIRST 3 RECORDS
print("\n--- First 3 Records ---")
print(titanic_daylin.head(3))

#PRINT THE SAHPE OF THE DATAFRAME
print("\n--- Shape of DataFrame ---")
print(titanic_daylin.shape)

#PRINT NAMES, TYPES AND COUNTS SHOWING MISSING VALUES PER COLUMNS
print("\n--- Column Data Types ---")
print(titanic_daylin.dtypes)

print("\n--- DataFrame Info ---")
titanic_daylin.info()

print("\n--- Missing Values Per Column ---")
print(titanic_daylin.isnull().sum())

#PRINT UNIQUE VALUES FOR SEX AND PCLASS
print("\n--- Unique values in Sex column ---")
print(titanic_daylin["Sex"].unique())

print("\n--- Unique values in Pclass column ---")
print(titanic_daylin["Pclass"].unique())


#DATA VISUALIZATION
#Survived vs Passanger Classs
survival_by_class = pd.crosstab(titanic_daylin['Pclass'], titanic_daylin['Survived'])

print("\n---Survived vs Passenger Class ---")
print(survival_by_class)

survival_by_class.plot(kind='bar')
plt.xlabel("Passenger Class")
plt.ylabel("Number of Passengers")
plt.title("Survival Count by Passenger Class - Daylin")
plt.legend(title="Survived (0 = No, 1 = Yes)")
plt.show()

#Survived vs Sex
survival_by_sex = pd.crosstab(titanic_daylin['Sex'], titanic_daylin['Survived'])

print("\n--- Survived vs Sex ---")
print(survival_by_sex)

survival_by_sex.plot(kind='bar')
plt.xlabel("Gender")
plt.ylabel("Number of Passengers")
plt.title("Survival Count by Gender - Daylin")
plt.legend(title="Survived (0 = No, 1 = Yes)")
plt.show()

#Scatter matrix plot 
#1st everything has to be numeric 
titanic_daylin['Sex'] = titanic_daylin['Sex'].map({'female': 0, 'male': 1})

#2nd select columns asked 
features = titanic_daylin[['Survived', 'Sex', 'Pclass', 'Fare', 'SibSp', 'Parch']]

#Display the matrix plot
print("\n--- Scatter Matrix Plot ---")
scatter_matrix(features, alpha=0.5, figsize=(10, 10), diagonal='hist')
plt.suptitle("Scatter Matrix of Survival Factors - Daylin", fontsize=16)
plt.show()

#DATA TRANSFORMATION
#Drop columns cabin, name, passengerID, ticket
titanic_daylin = titanic_daylin.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
print(titanic_daylin.head()) #this is just to confirm that they were drop
print(titanic_daylin.shape)

#Get dummies, transform categorical values in df into numeric values
print("\n--- Get dummies ---")
titanic_daylin = pd.get_dummies(titanic_daylin, columns=['Embarked'], drop_first=True)

titanic_daylin[['Embarked_Q', 'Embarked_S']] = titanic_daylin[['Embarked_Q', 'Embarked_S']].astype(int)

print(titanic_daylin.head())
print(titanic_daylin.shape)

#show the created variables and drop original ones
print(titanic_daylin.columns)

#calculate the mean of age 
titanic_daylin['Age'].mean()
#print("Mean of Age:", titanic_daylin['Age'].mean())

#change the NaN values for the mean of the age
titanic_daylin['Age'] = titanic_daylin['Age'].fillna(titanic_daylin['Age'].mean())
print("\n--- Change the NaN values for the mean of the Age ---")
print(titanic_daylin.head())
print(titanic_daylin.shape)

#convert everything to floar
titanic_daylin = titanic_daylin.astype(float)
print("\n--- Data Types After Converting to Float ---")
print(titanic_daylin.dtypes)

# Show dataframe structure (This is step #7)
print("\n--- DataFrame Info (After Conversion) ---")
titanic_daylin.info()

#use the formula
def normalize_df(df):
    return (df - df.min()) / (df.max() - df.min())
titanic_daylin = normalize_df(titanic_daylin)
print(titanic_daylin.head(2))#Display print the first 2 records 
#print(titanic_daylin.describe())

#generate image 9 by 10 inches
titanic_daylin.hist(figsize=(9,10))
plt.show()

#SPLIT FEATURES X AND Y
# Split features (X) and target (Y)
x_daylin = titanic_daylin.drop('Survived', axis=1)
y_daylin = titanic_daylin['Survived']

print("\n--- X DataFrame (Features) ---")
print(x_daylin.head())

print("\n--- Y DataFrame (Target) ---")
print(y_daylin.head())

#TRAIN SPLIT AND TEST SPLTT
# Split data (70% train, 30% test) using random_state = 10
x_train_daylin, x_test_daylin, y_train_daylin, y_test_daylin = train_test_split(
    x_daylin, y_daylin, test_size=0.30, random_state=10)

print("\n--- Shapes ---")
print("x_train_daylin:", x_train_daylin.shape)
print("x_test_daylin:", x_test_daylin.shape)
print("y_train_daylin:", y_train_daylin.shape)
print("y_test_daylin:", y_test_daylin.shape)

#BUID & VALIDATE THE MODEL 
# Fit logistic regression to the training data
daylin_model = LogisticRegression(max_iter=1000)
daylin_model.fit(x_train_daylin, y_train_daylin)

print("\nModel trained: ", daylin_model)

# Display the coefficients of the logistic regression model
coeff_table = pd.DataFrame(
    list(zip(x_train_daylin.columns, np.transpose(daylin_model.coef_))),
    columns=['Feature', 'Coefficient']
)

print("\n--- Logistic Regression Coefficients ---")
print(coeff_table)

#CROSS VALIDATION 
# Test sizes from 10% to 50% increasing by 5%
test_sizes = np.round(np.arange(0.10, 0.51, 0.05), 2)

print("\n--- Cross Validation Results for Different Train/Test Splits ---")

for ts in test_sizes:
    # Split using last two digits of student ID = 10
    X_train_cv, X_test_cv, y_train_cv, y_test_cv = train_test_split(
        x_daylin, y_daylin, test_size=ts, random_state=10
    )
    
    # Create new logistic model for each split
    model_cv = LogisticRegression(max_iter=1000)
    
    # Perform 10 cross validation on the training portion only
    scores = cross_val_score(model_cv, X_train_cv, y_train_cv, cv=10)
    
    # Print min, mean, and max accuracy for this test size
    print(f"Test Size = {ts:.2f} -> Min: {scores.min():.4f}, Mean: {scores.mean():.4f}, Max: {scores.max():.4f}")
    

#TEST THE MODEL 
# REBUILD MODEL using the 70% - 30% split
daylin_model = LogisticRegression(max_iter=1000)
daylin_model.fit(x_train_daylin, y_train_daylin)

print("\nModel rebuilt using the 70/30 training data split.")

#Get predicted probabilities
y_pred_daylin = daylin_model.predict_proba(x_test_daylin)

print("\n--- Predicted Probabilities (y_pred_daylin) ---")
print(y_pred_daylin)  # show first 5 predictions

# Convert probabilities to boolean classification using threshold = 0.5
y_pred_daylin_flag = y_pred_daylin[:, 1] > 0.5

print("\n--- y_pred_daylin_flag (Predicted Class as True or False) ---")
print(y_pred_daylin_flag)  # opcional: muestra primeros 10 resultados

# Compute model accuracy on the TEST data
test_accuracy = accuracy_score(y_test_daylin, y_pred_daylin_flag)

print("\n--- Model Accuracy on Test Data ---")
print(test_accuracy)

print("\n--- Confusion Matrix ---")
print(confusion_matrix(y_test_daylin, y_pred_daylin_flag))

print("\n--- Classification Report ---")
print(classification_report(y_test_daylin, y_pred_daylin_flag))

#Change threshold to 0.75
y_pred_daylin_flag_075 = y_pred_daylin[:, 1] > 0.75

print("\n--- Predictions with Threshold = 0.75 ---")
print(y_pred_daylin_flag_075[:10])

print("\n--- Accuracy with Threshold = 0.75 ---")
test_accuracy_075 = accuracy_score(y_test_daylin, y_pred_daylin_flag_075)
print(test_accuracy_075)

print("\n--- Confusion Matrix (Threshold = 0.75) ---")
print(confusion_matrix(y_test_daylin, y_pred_daylin_flag_075))

print("\n--- Classification Report (Threshold = 0.75) ---")
print(classification_report(y_test_daylin, y_pred_daylin_flag_075))

# Predictions on training data using threshold = 0.75
y_train_pred_proba = daylin_model.predict_proba(x_train_daylin)
y_train_pred_flag_075 = y_train_pred_proba[:, 1] > 0.75

print("\n--- Training Accuracy (Threshold = 0.75) ---")
train_accuracy_075 = accuracy_score(y_train_daylin, y_train_pred_flag_075)
print(train_accuracy_075)

# --- Training accuracy with threshold = 0.50 (to compare with test) ---
y_train_pred_flag_05 = daylin_model.predict_proba(x_train_daylin)[:, 1] > 0.50
train_accuracy_05 = accuracy_score(y_train_daylin, y_train_pred_flag_05)

print("\n--- Training Accuracy (Threshold = 0.50) ---")
print(train_accuracy_05)

# --- Resumen comparativo (opcional pero útil para tu reporte) ---
print("\n=== Accuracy Comparison ===")
print(f"Test  (thr=0.50): {test_accuracy:.4f}")
print(f"Train (thr=0.50): {train_accuracy_05:.4f}")
print(f"Test  (thr=0.75): {test_accuracy_075:.4f}")
print(f"Train (thr=0.75): {train_accuracy_075:.4f}")

