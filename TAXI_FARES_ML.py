import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

#LOAD DATA

df = pd.read_csv(r"C:\Users\George\Desktop\Data-Programming\projects for upload\Machine Learning\TAXI FARES\Taxi_Trip_Data_preprocessed.csv")

# print(df.head())
# print(df.isnull().sum())

#DATA PREPROCESSING

df['payment_type'] = df['payment_type'].map({'Card': 1, 'Cash':2 })

#DROP COLUMNS WITH MISSING VALUES

df = df.dropna()

# # DEFINE X / Y

X = df.drop('fare_amount', axis=1)
Y = df['fare_amount']

# TRAIN TEST SPLIT

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=42)


# MODEL

model = RandomForestRegressor(random_state=42)
model.fit(X_train, Y_train)

# PREDICTIONS
Y_pred = model.predict(X_test)

# EVALUATION

mse = mean_squared_error(Y_test, Y_pred)
r2 = r2_score(Y_test, Y_pred)

print("MSE:", mse)
print("R2:", r2)


# VISUALIZATION

plt.figure(figsize=(8,6))
plt.scatter(Y_test, Y_pred)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted")

plt.plot([Y_test.min(), Y_test.max()],
         [Y_test.min(), Y_test.max()],
         color='red')

plt.show()