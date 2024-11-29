import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Set the number of CSV files to import
NUM_FILES = 5  # Change this to the number of files you want to import

# Load all CSV files from the "dataset" folder
def load_data(num_files):
    all_data = []
    for i in range(num_files):
        file_path = f'dataset/data_log_{i}.csv'
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            all_data.append(df)
        else:
            print(f"File not found: {file_path}")
    combined_data = pd.concat(all_data, ignore_index=True)
    return combined_data

# Load data
data = load_data(NUM_FILES)

# Preprocess data
# Drop rows with missing values
data = data.dropna()

# Extract features (X) and target (y)
X = data.drop(columns=['nivel_rio'])
y = data['nivel_rio']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

### 1. Linear Regression
print("Training Linear Regression model...")
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_predictions = lr_model.predict(X_test)

# Evaluation
lr_mse = mean_squared_error(y_test, lr_predictions)
lr_mae = mean_absolute_error(y_test, lr_predictions)
print(f"Linear Regression - MSE: {lr_mse}, MAE: {lr_mae}")

### 2. Random Forest
print("Training Random Forest model...")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)

# Evaluation
rf_mse = mean_squared_error(y_test, rf_predictions)
rf_mae = mean_absolute_error(y_test, rf_predictions)
print(f"Random Forest - MSE: {rf_mse}, MAE: {rf_mae}")

### 3. LSTM (Long Short-Term Memory)
# Prepare data for LSTM
X_train_lstm = np.expand_dims(X_train.values, axis=1)
X_test_lstm = np.expand_dims(X_test.values, axis=1)

print("Training LSTM model...")
lstm_model = Sequential([
    LSTM(64, activation='relu', input_shape=(1, X_train.shape[1])),
    Dense(1)
])
lstm_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
lstm_model.fit(X_train_lstm, y_train, epochs=20, batch_size=32, verbose=1)

# Make predictions with LSTM
lstm_predictions = lstm_model.predict(X_test_lstm)

# Evaluation
lstm_mse = mean_squared_error(y_test, lstm_predictions)
lstm_mae = mean_absolute_error(y_test, lstm_predictions)
print(f"LSTM - MSE: {lstm_mse}, MAE: {lstm_mae}")

### Plot Predictions
plt.figure(figsize=(15, 5))
plt.plot(y_test.values, label='Actual nivel_rio', color='blue')
plt.plot(lr_predictions, label='Linear Regression Predictions', color='green')
plt.plot(rf_predictions, label='Random Forest Predictions', color='orange')
plt.plot(lstm_predictions, label='LSTM Predictions', color='red')
plt.legend()
plt.title('Model Predictions vs Actual')
plt.xlabel('Sample Index')
plt.ylabel('nivel_rio')
plt.show()
