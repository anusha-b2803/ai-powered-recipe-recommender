import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def get_clustering_pipeline(n_clusters=5):
    """
    Returns a Pipeline containing Data Scaling/Encoding and KMeans.
    Features used: Cuisine_Type, Cooking_Time_Minutes, Calories_Per_Serving, Popularity_Score, Difficulty_Score
    """
    
    # Define which columns are numeric vs categorical
    numeric_features = ['Cooking_Time_Minutes', 'Calories_Per_Serving', 'Popularity_Score', 'Difficulty_Score']
    categorical_features = ['Cuisine_Type']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('clusterer', KMeans(n_clusters=n_clusters, random_state=42, n_init=10))
    ])
    
    return pipeline

def evaluate_clustering(pipeline, df_recipes):
    """
    Computes silhouette score for the clusters.
    """
    print("Evaluating Clustering performance...")
    X_processed = pipeline.named_steps['preprocessor'].transform(df_recipes)
    labels = pipeline.named_steps['clusterer'].labels_
    
    # Calculate Silhouette
    if len(set(labels)) > 1:
        score = silhouette_score(X_processed, labels)
        print(f"Silhouette Score: {score:.3f}")
        return score
    else:
        print("Clustering yielded 0 or 1 clusters.")
        return -1
