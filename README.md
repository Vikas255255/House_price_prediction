# House_price_prediction
# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Load the Boston Housing dataset
from sklearn.datasets import load_boston
boston_data = load_boston()
features = pd.DataFrame(boston_data.data, columns=boston_data.feature_names)
target = pd.DataFrame(boston_data.target, columns=['MEDV'])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Linear Regression Model
linear_model = LinearRegression()
linear_model.fit(X_train_scaled, y_train)

# Make predictions on the test set
linear_predictions = linear_model.predict(X_test_scaled)

# Evaluate the linear regression model
linear_rmse = np.sqrt(mean_squared_error(y_test, linear_predictions))
print("Linear Regression RMSE:", linear_rmse)

# Neural Network Model using TensorFlow and Keras
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=[X_train_scaled.shape[1]]),
    layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse')

# Train the neural network
model.fit(X_train_scaled, y_train, epochs=100, batch_size=32, verbose=0)

# Make predictions on the test set using the neural network
nn_predictions = model.predict(X_test_scaled)

# Evaluate the neural network model
nn_rmse = np.sqrt(mean_squared_error(y_test, nn_predictions))
print("Neural Network RMSE:", nn_rmse)
