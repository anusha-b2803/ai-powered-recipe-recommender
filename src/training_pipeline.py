import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split

from src.data_loader import load_data
from src.preprocessing import create_target, BasicCleaner
from src.feature_engineering import FeatureEngineer
from src.clustering_model import get_clustering_pipeline, evaluate_clustering
from src.classification_model import get_classification_pipeline, tune_classification_model

def run_training_pipeline():
    print("=== Training Pipeline Started ===")
    
    # 1. Load Data
    raw_path = r'a:\MP\DA MP\recipe-recommender\data\raw\recipes_data.csv'
    df = load_data(raw_path)
    
    # 2. Add Target
    df = create_target(df)
    
    # 3. Basic Cleaning
    cleaner = BasicCleaner()
    df = cleaner.transform(df)
    
    # 4. Feature Engineering
    fe = FeatureEngineer()
    df = fe.transform(df)
    
    # 5. Fit Clustering (on unique recipes)
    print("\n--- Fitting Clustering Model ---")
    df_recipes = df.drop_duplicates(subset=['Recipe_ID']).copy()
    clustering_pipe = get_clustering_pipeline(n_clusters=6)
    clustering_pipe.fit(df_recipes)
    evaluate_clustering(clustering_pipe, df_recipes)
    
    # 6. Fit Classification
    print("\n--- Fitting Classification Model ---")
    X = df # Features are extracted from this inside column transformer
    y = df['Preference_Class']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    raw_pipe = get_classification_pipeline()
    best_classifier = tune_classification_model(raw_pipe, X_train, y_train)
    
    train_acc = best_classifier.score(X_train, y_train)
    test_acc = best_classifier.score(X_test, y_test)
    print(f"Train Accuracy: {train_acc:.3f}")
    print(f"Test Accuracy: {test_acc:.3f}")
    
    # 7. Model Persistence
    models_dir = r'a:\MP\DA MP\recipe-recommender\models'
    os.makedirs(models_dir, exist_ok=True)
    
    joblib.dump(clustering_pipe, os.path.join(models_dir, 'clustering_model.pkl'))
    joblib.dump(best_classifier, os.path.join(models_dir, 'classifier_model.pkl'))
    print(f"Models saved successfully to {models_dir}")
    
    # Save processed dataframe for recommendations dashboard use
    processed_dir = r'a:\MP\DA MP\recipe-recommender\data\processed'
    os.makedirs(processed_dir, exist_ok=True)
    df.to_csv(os.path.join(processed_dir, 'processed_recipes.csv'), index=False)
    
    return best_classifier, clustering_pipe

if __name__ == "__main__":
    run_training_pipeline()
