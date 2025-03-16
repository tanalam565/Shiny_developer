import pandas as pd

def load_data(file_path):
    return pd.read_csv(file_path)

def clean_data(data):
    # Example function to clean the data
    data = data.dropna()  # Remove rows with missing values
    return data
