# -*- coding: utf-8 -*-
"""
Created on Wed Oct 15 20:12:09 2025

@author: dayli
"""


#GET THE DATA
import pandas as pd
import os
import matplotlib.pyplot as plt #parte 5

print("\n========== SECTION A: GET THE DATA ==========")

path = r"C:\Users\dayli\OneDrive\Escritorio\semester3\introductionai\Linear Regression\exercise2"
filename = 'Ecom Expense.csv'
fullpath = os.path.join(path,filename)
ecom_exp_daylin = pd.read_csv(fullpath)
print("Data successfully loaded into dataframe: ecom_exp_daylin\n")

#-------------------------------------------------

print("\n========== SECTION B: INITIAL EXPLORATION ==========")

# INITIAL EXPLORATION
###Check the data
#print(ecom_exp_daylin.columns.values)
#print(ecom_exp_daylin.describe)
#print(ecom_exp_daylin.dtypes)
#print(ecom_exp_daylin.info)
print("\n--- First 3 Records ---")
print(ecom_exp_daylin.head(3)) #First 3 records 

print("\n--- Shape of DataFrame ---")
print(ecom_exp_daylin.shape) #Print the shape of the data frame

print("\n--- Column Data Types ---")
print(ecom_exp_daylin.dtypes) # este imprime nombres de columnas y los typos de valores
#print(ecom_exp_daylin.isnull().sum())


print("\n--- Missing Values per Column ---")
missing_values = ecom_exp_daylin.isnull().sum()

for column, count in missing_values.items():
    if count > 0:
        print(f"{column}: {count}")
    else:
        print(f"{column}: 0")

#linea 23 y la 25 a la 31 es lo mismo e imprime lo mismo 

#--------------------------------------------------


# DATA TRANSFORMATION

#1) Get dummies

#valuetonumber_ecom_exp_daylin = pd.get_dummies(ecom_exp_daylin) #imprime los valores de golpe con su modificacion
#print(valuetonumber_ecom_exp_daylin)
print("\n========== SECTION C: DATA TRANSFORMATION ==========")

print("\n--- Step 1: Get Dummies ---")

transactionid_ecom_exp_daylin = pd.get_dummies(ecom_exp_daylin['Transaction ID'])
print(transactionid_ecom_exp_daylin)


#age_ecom_exp_daylin = pd.get_dummies(ecom_exp_daylin['Age']) #este ya es numero asi q no cambia 
#print(age_ecom_exp_daylin)


#items_ecom_exp_daylin = pd.get_dummies(ecom_exp_daylin['Items']) #este ya es numero asi q no cambia 
#print(items_ecom_exp_daylin)


monthlyincome_ecom_exp_daylin = pd.get_dummies(ecom_exp_daylin['Monthly Income']) #este ya es numero asi q no cambia 
print(monthlyincome_ecom_exp_daylin)


transactiontime_ecom_exp_daylin = pd.get_dummies(ecom_exp_daylin['Transaction Time']) #este ya es numero asi q no cambia 
print(transactiontime_ecom_exp_daylin) #float son numneros con decimales


record_ecom_exp_daylin = pd.get_dummies(ecom_exp_daylin['Record']) #este ya es numero asi q no cambia 
print(record_ecom_exp_daylin)


gender_ecom_exp_daylin = pd.get_dummies(ecom_exp_daylin['Gender']) 
print(gender_ecom_exp_daylin)

citytier_ecom_exp_daylin = pd.get_dummies(ecom_exp_daylin['City Tier'])
print(citytier_ecom_exp_daylin)


totalspend_ecom_exp_daylin = pd.get_dummies(ecom_exp_daylin['Total Spend']) #este ya es numero asi q no cambia 
print(totalspend_ecom_exp_daylin)

#get dummies convierte todas las variables categóricas 
#(de texto o tipo “object”) en columnas numéricas binarias (0 o 1).

#esta parte esta siendo separada por categoria unica de tier 
#sale true or false en algunas pq la columna original ya eran tipo booleano antes del get dummie ser aplicado

#La función pd.get_dummies():
#Convierte columnas categóricas (texto o tipo object) en columnas numéricas (0 o 1).
#Pero no cambia las columnas que ya son numéricas o booleanas, las deja igual.

#2drop original columns
print("\n--- Step 2: Drop Original Columns ---")
drop = ecom_exp_daylin.drop(columns=['Transaction ID', 'Gender', 'City Tier'])
print(drop)


#3  Attatch newly created variable to your dataframe
print("\n--- Step 3: Attach Newly Created Variables ---")
ecom_exp_daylin_final = pd.concat([drop,transactionid_ecom_exp_daylin, gender_ecom_exp_daylin, citytier_ecom_exp_daylin], axis=1)
# Convertir True/False a 1/0 antes de normalizar
ecom_exp_daylin_final = ecom_exp_daylin_final.astype(int)

#axis = 1  tiene como columan osea hace el attacth de la scolumnas
print(ecom_exp_daylin_final)

#4 Function that accepts a dataframe as argument an normalizes all data points
print("\n--- Step 4: Normalize the Data ---")
def normalization_ecom_exp_daylin(df):
    # Aplicar la fórmula de normalización (x - min) / (max - min)
    normalized_df = (df - df.min()) / (df.max() - df.min())
    return normalized_df

# Llamar la función pasando el dataframe final
normalized_ecom_exp_daylin = normalization_ecom_exp_daylin(ecom_exp_daylin_final)

# Limpiar espacios en nombres de columnas
normalized_ecom_exp_daylin.columns = normalized_ecom_exp_daylin.columns.str.strip()


# Imprimir los primeros dos records
print(normalized_ecom_exp_daylin.head(2))

#5 Pandas .hist to generate a plot showing all variable histograms 
#fig size 9 by 10 
print("\n--- Step 5: Generate Histograms ---")
# Seleccionar solo las columnas numéricas principales
#selected_columns = ['Age', 'Items', 'Monthly Income', 'Transaction Time', 'Record', 'Total Spend']

# Crear histograma de todas las columnas
normalized_ecom_exp_daylin.hist(figsize=(9, 10))

# Mostrar el gráfico
plt.show()

#6 Generate scatter matrix plot to illustrate relationships between selected variables
print("\n--- Step 6: Scatter Matrix Plot ---")
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt

# Seleccionar las columnas que pide la instrucción
selected_columns = ['Age', 'Monthly Income', 'Transaction Time', 'Total Spend']

# Crear la matriz de dispersión con los siguientes parámetros:
# - alpha=0.4: establece el nivel de transparencia de los puntos (para ver la densidad)
# - figsize=(13, 15): define el tamaño de la figura en pulgadas (ancho x alto)
# Cada gráfico de dispersión muestra cómo se relacionan dos variables,
# mientras que la diagonal contiene los histogramas de cada variable individual.

scatter_matrix(normalized_ecom_exp_daylin[selected_columns], 
               alpha=0.4, 
               figsize=(13, 15))
              

# Mostrar el gráfico
plt.show()

# El propósito del scatter matrix es identificar posibles relaciones o correlaciones entre las variables numéricas.

#--------------------------------------------------

#BUILD A MODEL 
print("\n==================== PART D: BUILD A MODEL ====================")

# En esta sección vamos a construir un modelo de regresión lineal
# Asumimos que existe una relación lineal entre la variable de salida (Total Spend)
# y las variables predictoras: Monthly Income, Transaction Time,
# y las variables dummy creadas anteriormente (Gender y City Tier).

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# 1) Definir las variables independientes (X) y la variable dependiente (y)
# y = Total Spend (variable objetivo o label)
# X = Monthly Income, Transaction Time, Gender (dummy) y City Tier (dummy)
print("\n--- Step 1: Define Independent and Dependent Variables ---")


# Seleccionamos las columnas necesarias
X_ecom_exp_daylin = normalized_ecom_exp_daylin[['Monthly Income', 'Transaction Time', 
                                                'Female', 'Male', 'Tier 1', 'Tier 2', 'Tier 3']]

y_ecom_exp_daylin = normalized_ecom_exp_daylin['Total Spend']

# 2) Split the data into training (65%) and testing (35%)
print("\n--- Step 2: Split Data (65% Training / 35% Testing) ---")
# Set the seed to be the last two digits of the student number (10)
X_train_daylin, X_test_daylin, y_train_daylin, y_test_daylin = train_test_split(
    X_ecom_exp_daylin, y_ecom_exp_daylin, test_size=0.35, random_state=10)

# 3) Store the training and testing data in DataFrames with the required names
print("\n--- Step 3: Store DataFrames ---")
x_train_daylin = pd.DataFrame(X_train_daylin)
y_train_daylin = pd.DataFrame(y_train_daylin)
x_test_daylin = pd.DataFrame(X_test_daylin)
y_test_daylin = pd.DataFrame(y_test_daylin)

# Print to confirm the data has been stored correctly
print("\n--- TRAINING AND TESTING DATA STORED ---")
print("x_train_daylin shape:", x_train_daylin.shape)
print("y_train_daylin shape:", y_train_daylin.shape)
print("x_test_daylin shape:", x_test_daylin.shape)
print("y_test_daylin shape:", y_test_daylin.shape)

# 4) Create the linear regression model
print("\n--- Step 4: Create Linear Regression Model ---")
linear_model_daylin = LinearRegression() #linearregression crear el modelo vacio

# 5) Train the model using the training data
print("\n--- Step 5: Train the Model ---")
linear_model_daylin.fit(x_train_daylin, y_train_daylin) #.fit ... lo entrena usando datos de entrenamiento 
print("\nLinear Regression model successfully trained with training data.")

# Print confirmation for fit in train



# 6) Make predictions using the testing data
print("\n--- Step 6: Make Predictions ---")
y_pred_daylin = linear_model_daylin.predict(x_test_daylin)

# 7) Evaluate the model using the R² score
print("\n--- Step 7: Evaluate Model ---")
r2_daylin = r2_score(y_test_daylin, y_pred_daylin)

# 8) Display the model results
print("\n--- LINEAR REGRESSION MODEL RESULTS ---")
print("Intercept (b0):", linear_model_daylin.intercept_)
print("Coefficients (b1...bn):", linear_model_daylin.coef_)
print("R² Score:", round(r2_daylin, 4))

# 9) Display the first actual vs predicted values
# Convert both actual and predicted values to 1D arrays
actual_values = y_test_daylin.values.flatten()
predicted_values = y_pred_daylin.flatten()

comparison_daylin = pd.DataFrame({
    'Actual': actual_values[:10],
    'Predicted': predicted_values[:10]
})

print(comparison_daylin)

# 10) Define the new predictors including 'Record'
print("\n--- Step 10: Define New Predictors Including 'Record' ---")
X_ecom_exp_daylin_record = normalized_ecom_exp_daylin[['Monthly Income', 'Transaction Time', 
                                                       'Female', 'Male', 'Tier 1', 'Tier 2', 'Tier 3', 'Record']]
y_ecom_exp_daylin_record = normalized_ecom_exp_daylin['Total Spend']

# 11) Split the data again (65% training, 35% testing)
print("\n--- Step 11: Split Data Again (With 'Record') ---")
x_train_daylin_record, x_test_daylin_record, y_train_daylin_record, y_test_daylin_record = train_test_split(
    X_ecom_exp_daylin_record, y_ecom_exp_daylin_record, test_size=0.35, random_state=10)

# 12) Train a new linear regression model
print("\n--- Step 12: Train New Model ---")
linear_model_daylin_record = LinearRegression()
linear_model_daylin_record.fit(x_train_daylin_record, y_train_daylin_record)

# 13) Make predictions
print("\n--- Step 13: Make Predictions for New Model ---")
y_pred_daylin_record = linear_model_daylin_record.predict(x_test_daylin_record)

# 14) Evaluate the new model (with 'Record')
print("\n--- Step 14: Evaluate New Model ---")
r2_daylin_record = r2_score(y_test_daylin_record, y_pred_daylin_record)

# 15) Display the coefficients and R² score
print("\n--- LINEAR REGRESSION MODEL (INCLUDING 'RECORD') ---")
print("Intercept (b0):", linear_model_daylin_record.intercept_)
print("Coefficients (b1...bn):", linear_model_daylin_record.coef_)
print("R² Score:", round(r2_daylin_record, 4))

# Explicacion general
# El modelo de regresión lineal intenta ajustar una línea (o plano en múltiples dimensiones)
# que mejor explica la relación entre las variables independientes (X) y la variable dependiente (y).
# El R² score indica qué tan bien el modelo explica la variabilidad de los datos.
# Un R² cercano a 1 significa un mejor ajuste del modelo.
