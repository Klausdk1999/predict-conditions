import pandas as pd

# Load data
file_path = 'dataset/data_log_0.csv'
data = pd.read_csv(file_path)

# Display basic information about the dataset
print("Columns in the dataset:", data.columns)
print("\n")
print("First few rows of the dataset:")
print(data.head())
