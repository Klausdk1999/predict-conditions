import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load data
data = pd.read_csv('dataset/data_log_0.csv')

# Prepare features and target
X = data[['rain_level', 'other_relevant_features']]  # Features
y = data['river_level']  # Target

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Example prediction for the next 24 hours
next_day_rain = [[current_rain_level, other_features]]  # Replace with actual values
next_day_prediction = model.predict(next_day_rain)

print(f"Predicted river level for the next 24 hours: {next_day_prediction[0]}")
