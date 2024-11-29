import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import os

# Configuration
DATA_FOLDER = './dataset/'  # Path to your dataset folder
FILE_PREFIX = 'data_log_'
NUM_FILES_TO_LOAD = 10  # Set how many files to load at once
TARGET_COLUMN = 'nivel_rio'
FEATURE_COLUMNS = ['chuva_001h', 'chuva_003h', 'chuva_006h', 'chuva_012h', 
                   'chuva_024h', 'chuva_048h', 'chuva_072h', 'chuva_096h',
                   'chuva_120h', 'chuva_144h', 'chuva_168h', 'temp_atual', 
                   'umidade', 'vel_vento', 'pres_atmos']  # Adjust as needed

def load_and_preprocess_data(data_folder, file_prefix, num_files_to_load):
    """Loads and preprocesses data from multiple CSV files."""
    all_data = pd.DataFrame()

    # Load data from multiple files
    for i in range(num_files_to_load):
        file_path = os.path.join(data_folder, f"{file_prefix}{i}.csv")
        if os.path.exists(file_path):
            print(f"Loading file: {file_path}")
            data = pd.read_csv(file_path)
            
            # Print columns in the file to debug
            print(f"Columns in file {file_path}: {data.columns.tolist()}")
            
            # Filter necessary columns and drop rows with missing values
            missing_columns = [col for col in FEATURE_COLUMNS + [TARGET_COLUMN] if col not in data.columns]
            if missing_columns:
                print(f"Missing columns in file {file_path}: {missing_columns}")
            else:
                data = data[FEATURE_COLUMNS + [TARGET_COLUMN]]
                data = data.dropna()

                # Append to main DataFrame
                all_data = pd.concat([all_data, data], ignore_index=True)
        else:
            print(f"File not found: {file_path}")

    return all_data

def main():
    # Load and preprocess data
    data = load_and_preprocess_data(DATA_FOLDER, FILE_PREFIX, NUM_FILES_TO_LOAD)

    if data.empty:
        print("No valid data loaded. Please check the dataset.")
        return

    # Prepare features and target
    X = data[FEATURE_COLUMNS]
    y = data[TARGET_COLUMN]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluate the model
    score = model.score(X_test, y_test)
    print(f"Model R^2 score: {score:.2f}")

    # Example prediction for the next 24 hours
    # Replace with actual input values
    next_day_input = np.array([[5.2, 10.1, 15.3, 20.4, 25.0, 30.0, 35.5, 40.2, 45.0, 50.0, 55.0, 22.0, 80.0, 5.0, 1013.0]])  
    next_day_prediction = model.predict(next_day_input)

    print(f"Predicted river level for the next 24 hours: {next_day_prediction[0]:.2f}")

if __name__ == '__main__':
    main()
