import pandas as pd
import numpy as np
import joblib
import os

class RecipeRecommender:
    def __init__(self, models_dir=None, processed_data_path=None):
        # Dynamically evaluate absolute paths relative to execution folder context
        if models_dir is None:
            models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
        if processed_data_path is None:
            processed_data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'processed', 'processed_recipes.csv')
            
        self.models_dir = models_dir
        self.processed_data_path = processed_data_path
        self.classifier = None
        self.clusterer_pipe = None
        self.df = None
        self.load_models_and_data()
        
    def load_models_and_data(self):
        classifier_path = os.path.join(self.models_dir, 'classifier_model.pkl')
        clusterer_path = os.path.join(self.models_dir, 'clustering_model.pkl')
        
        if os.path.exists(classifier_path) and os.path.exists(clusterer_path):
            self.classifier = joblib.load(classifier_path)
            self.clusterer_pipe = joblib.load(clusterer_path)
            print("Completed loading saved models.")
        else:
            print("Models not found! Please run training_pipeline.py first.")
            
        if os.path.exists(self.processed_data_path):
            self.df = pd.read_csv(self.processed_data_path)
            # Add Cluster label column to the processed DF by running clusterer
            if self.clusterer_pipe is not None:
                # Deduplicate unique recipes
                df_recipes = self.df.drop_duplicates(subset=['Recipe_ID']).copy()
                preprocessor = self.clusterer_pipe.named_steps['preprocessor']
                X_clust = preprocessor.transform(df_recipes)
                clusters = self.clusterer_pipe.named_steps['clusterer'].labels_
                df_recipes['Cluster'] = clusters
                
                # Merge cluster back to full interactions or track separately
                self.recipe_clusters = df_recipes[['Recipe_ID', 'Cluster']]
                self.df = pd.merge(self.df, self.recipe_clusters, on='Recipe_ID', how='inner')
        else:
             print("Processed data not found! Please run training_pipeline.py first.")

    def recommend_recipes(self, user_filters, top_n=5):
        """
        user_filters: dict with keys:
            'Cuisine_Type', 'Cooking_Time_Minutes', 'Calories_Per_Serving', 'Difficulty_Score'
        """
        if self.classifier is None or self.clusterer_pipe is None:
            return pd.DataFrame(), 0, -1, False
            
        # Add default for clustering compatibility if missing
        if 'Popularity_Score' not in user_filters:
            user_filters['Popularity_Score'] = 5.0 # default average
            
        # 1. Map filters to clustering preprocessor input shape
        input_data = pd.DataFrame([user_filters])
        
        # 2. Predict closest cluster
        preprocessor = self.clusterer_pipe.named_steps['preprocessor']
        X_test = preprocessor.transform(input_data)
        nearest_cluster = self.clusterer_pipe.named_steps['clusterer'].predict(X_test)[0]
        
        print(f"Nearest Recipe Cluster found: {nearest_cluster}")
        
        # 3. Filter candidates in that cluster from the full unique recipes
        df_unique_recipes = self.df.drop_duplicates(subset=['Recipe_ID']).copy()
        matching_cluster_mask = self.df['Cluster'] == nearest_cluster
        recipe_ids = self.df.loc[matching_cluster_mask, 'Recipe_ID'].unique()
        
        df_candidates = df_unique_recipes[df_unique_recipes['Recipe_ID'].isin(recipe_ids)].copy()
        
        # 3. STRICT BOUNDS FILTERS: Ensure All conditions match perfectly
        df_candidates = df_candidates[df_candidates['Cuisine_Type'] == user_filters['Cuisine_Type']]
        
        # Apply ALL strict limits
        strict_mask = (df_candidates['Cooking_Time_Minutes'] <= user_filters.get('Cooking_Time_Minutes', 150)) & \
                      (df_candidates['Calories_Per_Serving'] <= user_filters.get('Calories_Per_Serving', 800)) & \
                      (df_candidates['Difficulty_Score'] == user_filters.get('Difficulty_Score', 2))
                      
        df_strict = df_candidates[strict_mask].copy()
        fallback_flag = False
        
        # If we have strict matches, use them! Otherwise, fallback to cluster candidates keeping Cuisine
        if not df_strict.empty:
            df_candidates = df_strict
        else:
            print("No candidates inside cluster matched all strict bounds. Falling back to relaxed cluster/cuisine pool.")
            fallback_flag = True
            # At minimum, keep the Cuisine strict
            if df_candidates.empty:
                df_candidates = df_unique_recipes[df_unique_recipes['Cuisine_Type'] == user_filters['Cuisine_Type']].copy()
            
        # 4. Score candidates with Decision Tree Classifier "High" probability
        # Create full feature matrix with a static synthetic User Profile for variables we don't have
        df_candidates['User_ID'] = 'U_TEMP'
        df_candidates['User_Preferences'] = user_filters.get('Cuisine_Type', "")
        df_candidates['Season'] = user_filters.get('Season', 'Summer')
        df_candidates['Occasion'] = user_filters.get('Occasion', 'Weeknight')
        
        # We need the full set of classification features
        # DecisionTreeClassifier predict_proba returns [Low, Medium, High] depending on target alphabetic order
        # Let's see the classes_
        classes = self.classifier.classes_
        high_idx = np.where(classes == 'High')[0]
        if len(high_idx) == 0:
            high_idx = [0] # fallback
        else:
            high_idx = high_idx[0]
            
        probs = self.classifier.predict_proba(df_candidates)
        df_candidates['Liking_Probability'] = probs[:, high_idx]
        
        # 5. Popularity-First Ranking: Sort strictly by Popularity_Score then Likability
        df_candidates = df_candidates.sort_values(by=['Popularity_Score', 'Liking_Probability'], ascending=[False, False])
        return df_candidates.head(top_n), len(df_candidates), nearest_cluster, fallback_flag

if __name__ == "__main__":
    # Test aggregator
    rec = RecipeRecommender()
    filters = {
        'Cuisine_Type': 'Italian',
        'Cooking_Time_Minutes': 25,
        'Calories_Per_Serving': 400,
        'Difficulty_Score': 1,
        'Season': 'Summer',
        'Occasion': 'Weeknight'
    }
    recommendations, _, _, _ = rec.recommend_recipes(filters)
    print(recommendations[['Recipe_Name', 'Cuisine_Type', 'Liking_Probability']])
