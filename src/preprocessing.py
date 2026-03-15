import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class BasicCleaner(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        X = X.copy()
        print("Running Basic Cleaning...")
        # Handle Imputation for critical numerics
        if 'Calories_Per_Serving' in X.columns:
            X['Calories_Per_Serving'] = X['Calories_Per_Serving'].fillna(X['Calories_Per_Serving'].median())
            
        if 'Serving_Size' in X.columns:
            X['Serving_Size'] = X['Serving_Size'].fillna(X['Serving_Size'].mode()[0])
            
        if 'Cooking_Time_Minutes' in X.columns:
            X['Cooking_Time_Minutes'] = X['Cooking_Time_Minutes'].fillna(X['Cooking_Time_Minutes'].median())
            
        # Clean Text columns
        text_cols = ['Ingredients_List', 'Preparation_Steps', 'User_Feedback', 'User_Preferences']
        for col in text_cols:
            if col in X.columns:
                X[col] = X[col].fillna("").astype(str).str.lower().str.strip()
                
        # Date conversion
        if 'Date_Prepared' in X.columns:
            X['Date_Prepared'] = pd.to_datetime(X['Date_Prepared'], errors='coerce')
            
        return X

def create_target(df):
    """
    Creates target variable for classification based on User_Rating.
    Thresholds:
    >= 4.0 -> High
    >= 3.0 and < 4.0 -> Medium
    < 3.0 -> Low
    """
    if 'User_Rating' not in df.columns:
        raise ValueError("User_Rating column not present in DataFrame for Target creation.")
        
    def get_preference(rating):
        if rating >= 4.0:
            return 'High'
        elif rating >= 3.0:
            return 'Medium'
        else:
            return 'Low'
            
    df = df.copy()
    df['Preference_Class'] = df['User_Rating'].apply(get_preference)
    # Target Class balance check output
    print("Target distribution:")
    print(df['Preference_Class'].value_counts(normalize=True))
    
    return df

if __name__ == "__main__":
    # Tests
    from src.data_loader import load_data
    df = load_data(r'a:\MP\DA MP\recipe-recommender\data\raw\recipes_data.csv')
    df = create_target(df)
    cleaner = BasicCleaner()
    df_clean = cleaner.transform(df)
    print(df_clean[['Ingredients_List', 'Preference_Class']].head(3))
