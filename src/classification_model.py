import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

def get_classification_pipeline():
    """
    Returns a Pipeline architecture for Classifying Preference Class.
    Includes text vectorization for ingredients and Categorical OneHot encoding.
    """
    
    numeric_features = ['Cooking_Time_Minutes', 'Calories_Per_Serving', 'Cost_Per_Serving', 'Popularity_Score', 'Ingredient_Count', 'Difficulty_Score', 'Is_Cuisine_Match', 'Is_Time_Match']
    categorical_features = ['Cuisine_Type', 'Season', 'Occasion']
    text_features = 'Ingredients_List' # Focus NLP on ingredients
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
            ('txt', TfidfVectorizer(max_features=50), text_features)
        ]
    )
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', DecisionTreeClassifier(random_state=42))
    ])
    
    return pipeline

def tune_classification_model(pipeline, X, y):
    """
    Runs GridSearchCV to optimize hyperparameters and prevent overfitting.
    """
    param_grid = {
        'classifier__max_depth': [3, 5, 8, 10, None],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 2, 4],
        'classifier__class_weight': ['balanced', None]
    }
    
    print("Starting Hyperparameter Tuning on Decision Tree...")
    grid_search = GridSearchCV(
        pipeline, 
        param_grid, 
        cv=5, 
        scoring='accuracy', 
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X, y)
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Best CV Accuracy: {grid_search.best_score_:.3f}")
    
    return grid_search.best_estimator_
