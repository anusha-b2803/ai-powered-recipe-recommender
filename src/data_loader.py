import pandas as pd
import os

def load_data(filepath):
    """
    Loads dataset from filepath, drops duplicates and provides basic logging.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found at {filepath}")
        
    df = pd.read_csv(filepath)
    print(f"Loaded dataset from {filepath}")
    print(f"Initial Shape: {df.shape}")
    
    # Remove duplicates
    initial_len = len(df)
    df = df.drop_duplicates()
    if len(df) < initial_len:
        print(f"Removed {initial_len - len(df)} duplicate rows.")
        
    return df

if __name__ == "__main__":
    # Test loading
    try:
        df = load_data(r'a:\MP\DA MP\recipe-recommender\data\raw\recipes_data.csv')
        print(df.head(2))
    except Exception as e:
        print(f"Error: {e}")
