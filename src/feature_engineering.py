import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        X = X.copy()
        print("Running Feature Engineering...")
        
        # 1. Cooking_Time_Category
        if 'Cooking_Time_Minutes' in X.columns:
            def get_time_cat(t):
                if t <= 30: return 'Fast'
                elif t <= 60: return 'Medium'
                else: return 'Slow'
                
            X['Cooking_Time_Category'] = X['Cooking_Time_Minutes'].apply(get_time_cat)
            
        # 2. Difficulty_Score
        if 'Difficulty_Level' in X.columns:
            diff_map = {'Easy': 1, 'Medium': 2, 'Hard': 3}
            X['Difficulty_Score'] = X['Difficulty_Level'].map(diff_map).fillna(1).astype(int)
            
        # 3. Ingredient_Count
        if 'Ingredients_List' in X.columns:
            X['Ingredient_Count'] = X['Ingredients_List'].fillna("").apply(lambda x: len([i for i in x.split(",") if i.strip()]))
            
        # 4. Seasonal Match (Optional helpful feature)
        if 'Season' in X.columns and 'Seasonal_Availability' in X.columns:
            X['Is_Seasonal_Match'] = (X['Season'] == X['Seasonal_Availability']).astype(int)
            
        # 4b. Cuisine Preference Match (Boosts Classifier with interactive relation)
        if 'Cuisine_Type' in X.columns and 'User_Preferences' in X.columns:
            X['Is_Cuisine_Match'] = X.apply(lambda row: 1 if str(row['Cuisine_Type']) in str(row['User_Preferences']) else 0, axis=1)
            
        # 4c. Time Preference Match
        if 'Cooking_Time_Minutes' in X.columns and 'User_Preferences' in X.columns:
            def check_time_match(row):
                pref = str(row['User_Preferences']).lower()
                t = row['Cooking_Time_Minutes']
                if 'time: short' in pref and t <= 30: return 1
                if 'time: long' in pref and t > 60: return 1
                return 0
            X['Is_Time_Match'] = X.apply(check_time_match, axis=1)
            
        # 5. Popularity_Level
        if 'Popularity_Score' in X.columns:
            X['Popularity_Level'] = pd.qcut(X['Popularity_Score'], q=3, labels=['Low', 'Medium', 'High']).astype(str)
            
        return X

if __name__ == "__main__":
    import pandas as pd
    # Quick Test DataFrame
    data = pd.DataFrame({
        'Cooking_Time_Minutes': [20, 45, 90],
        'Difficulty_Level': ['Easy', 'Medium', 'Hard'],
        'Ingredients_List': ['pasta, tomato, basil', 'rice, curry', 'beef, veggies, water, salt'],
        'Popularity_Score': [3.5, 7.8, 9.1],
        'Season': ['Summer', 'Winter', 'Spring'],
        'Seasonal_Availability': ['Summer', 'Summer', 'Spring']
    })
    
    fe = FeatureEngineer()
    engineered_df = fe.transform(data)
    print(engineered_df)
