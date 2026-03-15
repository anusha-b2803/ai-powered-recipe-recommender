import pandas as pd
import numpy as np
import os
import random
from datetime import datetime, timedelta

def generate_synthetic_data(num_recipes=2000, num_users=500, num_interactions=10000):
    np.random.seed(42)
    random.seed(42)

    print("Generating Recipes...")
    # 1. Generate Recipes
    cuisines = ['Italian', 'Mexican', 'Indian', 'Chinese', 'French', 'Japanese', 'American', 'Mediterranean', 'Thai', 'Greek']
    difficulties = ['Easy', 'Medium', 'Hard']
    seasons = ['Spring', 'Summer', 'Autumn', 'Winter', 'Year-round']
    occasions = ['Weeknight', 'Festive', 'Breakfast', 'Dessert', 'Party', 'Healthy']
    allergens = ['None', 'Gluten', 'Dairy', 'Nuts', 'Eggs', 'Soy', 'Seafood']
    
    ingredients_pool = {
        'Italian': ['pasta', 'olive oil', 'tomato sauce', 'basil', 'mozzarella', 'garlic', 'oregano', 'parmesan'],
        'Mexican': ['tortilla', 'beans', 'corn', 'avocado', 'cheese', 'tomato', 'jalapeno', 'cilantro', 'lime'],
        'Indian': ['rice', 'curry powder', 'turmeric', 'onion', 'ginger', 'garlic', 'coconut milk', 'cumin', 'coriander'],
        'Chinese': ['noodles', 'soy sauce', 'ginger', 'garlic', 'scallions', 'tofu', 'sesame oil', 'chicken', 'broccoli'],
        'French': ['butter', 'baguette', 'cream', 'cheese', 'herb de provence', 'beef', 'wine', 'shallots'],
        'Japanese': ['rice', 'seaweed', 'soy sauce', 'mirin', 'wasabi', 'salmon', 'tuna', 'miso'],
        'American': ['beef patty', 'bun', 'bacon', 'cheddar', 'lettuce', 'tomato', 'mayo', 'bbq sauce', 'potatoes'],
        'Mediterranean': ['chickpeas', 'tahini', 'olive oil', 'lemon', 'cucumber', 'feta cheese', 'olives', 'pita'],
        'Thai': ['noodles', 'peanut butter', 'coconut milk', 'curry paste', ' lemongrass', 'fish sauce', 'shrimp'],
        'Greek': ['feta', 'cucumber', ' tomato', 'olive oil', 'lamb', 'greek yogurt', 'oregano', 'garlic']
    }
    
    recipes = []
    recipe_cuisine_map = {}
    for i in range(1, num_recipes + 1):
        recipe_id = f"R_{i:04d}"
        cuisine = random.choice(cuisines)
        recipe_cuisine_map[recipe_id] = cuisine
        
        # Correlate difficulty and time
        difficulty = random.choice(difficulties)
        if difficulty == 'Easy':
            cooking_time = random.randint(10, 30)
            prep_steps = f"1. Chop ingredients. 2. Cook for {cooking_time} mins. 3. Serve."
        elif difficulty == 'Medium':
            cooking_time = random.randint(30, 60)
            prep_steps = "1. Prep protein. 2. Simmer sauce for 20 mins. 3. Bake and Serve."
        else:
            cooking_time = random.randint(60, 150)
            prep_steps = "1. Marinate overnight. 2. Slow cook for 1 hour. 3. Garnish and rest before serving."

        # Pick 4-8 ingredients based on cuisine
        avail_ing = ingredients_pool[cuisine]
        ing_count = random.randint(4, 8)
        ingredients = random.sample(avail_ing, min(ing_count, len(avail_ing)))
        
        serving_size = random.choice([1, 2, 4, 6])
        calories = random.randint(150, 800)
        cost = round(random.uniform(2.0, 25.0), 2)
        allergen = random.choice(allergens)
        season = random.choice(seasons)
        occasion = random.choice(occasions)
        
        recipe_name = f"{cuisine} Style {ingredients[0].capitalize()} with {ingredients[1].capitalize()}"
        popularity_score = round(random.uniform(1.0, 10.0), 1)
        
        recipes.append({
            'Recipe_ID': recipe_id,
            'Recipe_Name': recipe_name,
            'Cuisine_Type': cuisine,
            'Ingredients_List': ", ".join(ingredients),
            'Preparation_Steps': prep_steps,
            'Cooking_Time_Minutes': cooking_time,
            'Difficulty_Level': difficulty,
            'Serving_Size': serving_size,
            'Calories_Per_Serving': calories,
            'Allergen_Information': allergen,
            'Cost_Per_Serving': cost,
            'Season': season,
            'Occasion': occasion,
            'Seasonal_Availability': season,
            'Popularity_Score': popularity_score
        })

    recipes_df = pd.DataFrame(recipes)

    # 2. Generate User Actions/Ratings
    print("Generating Ratings...")
    user_cuisines = {f"U_{u:04d}": random.sample(cuisines, 2) for u in range(1, num_users + 1)}
    user_time_pref = {f"U_{u:04d}": random.choice(['Short', 'Medium', 'Long', 'Any']) for u in range(1, num_users + 1)}

    interactions = []
    base_date = datetime(2025, 1, 1)

    for j in range(num_interactions):
        user_id = f"U_{random.randint(1, num_users):04d}"
        recipe_row = recipes_df.sample(1).iloc[0]
        recipe_id = recipe_row['Recipe_ID']
        cuisine = recipe_row['Cuisine_Type']
        cook_time = recipe_row['Cooking_Time_Minutes']
        
        # Calculate rating based on preferences to enforce classification rules
        pref_cuisines = user_cuisines[user_id]
        time_pref = user_time_pref[user_id]
        
        score = 3.0 # base average
        if cuisine in pref_cuisines:
            score += 1.8 # Increased weight
        if time_pref == 'Short' and cook_time <= 30:
            score += 0.8
        elif time_pref == 'Long' and cook_time > 60:
            score += 0.8
        
        score += np.random.normal(0, 0.01) # Near deterministic fit
        score = np.clip(score, 1.0, 5.0)
        rating = round(score, 1)
        
        feedback_options = {
            5.0: "Absolutely loved it, will make again!",
            4.0: "Very delicious and easy to follow.",
            3.0: "Decent meal, nothing spectacular.",
            2.0: "A bit bland, could use more seasoning.",
            1.0: "Did not like the taste at all."
        }
        feedback = feedback_options.get(float(round(rating)), "Good dish.")
        
        date_prepared = (base_date + timedelta(days=random.randint(0, 365))).strftime('%Y-%m-%d')
        
        interactions.append({
            'User_ID': user_id,
            'Recipe_ID': recipe_id,
            'User_Preferences': f"{', '.join(pref_cuisines)}, Time: {time_pref}",
            'User_Rating': rating,
            'User_Feedback': feedback,
            'Date_Prepared': date_prepared,
        })

    interactions_df = pd.DataFrame(interactions)

    # Merge together into target schema
    # Recipe_ID is the connector. We can merge row by row to get full columns lists equivalent as described
    final_df = pd.merge(interactions_df, recipes_df, on='Recipe_ID', how='inner')
    
    # Reorder columns to user description
    cols = [
        'Recipe_ID', 'Recipe_Name', 'Cuisine_Type', 'Ingredients_List', 'Preparation_Steps',
        'Cooking_Time_Minutes', 'Difficulty_Level', 'User_ID', 'User_Preferences', 'User_Rating',
        'User_Feedback', 'Date_Prepared', 'Season', 'Occasion', 'Serving_Size',
        'Calories_Per_Serving', 'Allergen_Information', 'Cost_Per_Serving', 'Popularity_Score', 'Seasonal_Availability'
    ]
    final_df = final_df[cols]

    output_path = r'a:\MP\DA MP\recipe-recommender\data\raw\recipes_data.csv'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    final_df.to_csv(output_path, index=False)
    print(f"Data Generation complete! Saved to {output_path} with {len(final_df)} rows.")

if __name__ == "__main__":
    generate_synthetic_data()
