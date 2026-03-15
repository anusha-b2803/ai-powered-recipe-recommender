import streamlit as st
# Force Reload Trigger
import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Adjust system path to find 'src'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.ui_components import set_page_config, inject_custom_css, render_recipe_card
import importlib
import src.recommender
importlib.reload(src.recommender)
from src.recommender import RecipeRecommender

# 1. Initialize Page and CSS
set_page_config()
inject_custom_css()

# 2. Load Recommender Engine
def load_recommender():
    return RecipeRecommender()

rec_engine = load_recommender()

# 2b. Recipe Details Modal
@st.dialog("Recipe Details 🍽️")
def show_recipe_details(row):
    st.markdown(f"### {row['Recipe_Name']}")
    st.markdown(f"**🌍 Cuisine:** {row['Cuisine_Type']}")
    st.markdown(f"**⭐ Popularity Score:** {row['Popularity_Score']}/10")
    st.markdown(f"**💰 Estimated Cost:** ${row['Cost_Per_Serving']} per serving")
    
    st.markdown("---")
    st.markdown("#### 📋 Ingredients")
    ingredients = str(row['Ingredients_List']).split(",")
    for ing in ingredients:
        if ing.strip():
            st.markdown(f"- {ing.strip().capitalize()}")
            
    st.markdown("---")
    st.markdown("#### 🍳 Preparation Steps")
    # Clean steps by splitting index points if available or full stops
    steps = str(row['Preparation_Steps']).split(".")
    for idx, step in enumerate(steps):
        clean_step = step.strip()
        if clean_step and not clean_step.isdigit():
            # Remove leading numbers if they exist from generation
            if clean_step.startswith(tuple(str(i) for i in range(1,10))):
                clean_step = clean_step.split(" ", 1)[-1] if " " in clean_step else clean_step
            st.markdown(f"**{idx+1}.** {clean_step.capitalize()}")

# 3. Sidebar Filters
st.sidebar.markdown('<div class="sidebar-header">Preferences 🔍</div>', unsafe_allow_html=True)

cuisines = ['Italian', 'Mexican', 'Indian', 'Chinese', 'French', 'Japanese', 'American', 'Mediterranean', 'Thai', 'Greek']
selected_cuisine = st.sidebar.selectbox("Preferred Cuisine", options=cuisines)

prep_time = st.sidebar.slider("Maximum Preparation Time (min)", 10, 150, 45)
difficulty = st.sidebar.select_slider("Preferred Difficulty", options=['Easy', 'Medium', 'Hard'], value='Medium')
calories = st.sidebar.slider("Calorie limit (kcal)", 150, 800, 500)
season = st.sidebar.selectbox("Season", ['Spring', 'Summer', 'Autumn', 'Winter', 'Year-round'])
occasion = st.sidebar.selectbox("Occasion", ['Weeknight', 'Breakfast', 'Dessert', 'Festive', 'Party', 'Healthy'])

recommend_btn = st.sidebar.button("💡 Recommend Recipes")

# Main Title
st.markdown("<h1 style='text-align: center; color: #ff7f50;'>AI Powered Recipe Recommender</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #aaaaaa;'>Combining KMeans Clustering & Decision Tree Classification for personalized matches!</p>", unsafe_allow_html=True)

# 4. View Tabs
tab1, tab2 = st.tabs(["🎯 Recommendations", "📊 System Insights"])

with tab1:
    if recommend_btn or st.session_state.get('recommend_clicked', False):
        st.session_state['recommend_clicked'] = True
        with st.spinner("Finding best clusters & ranking..."):
            dif_score_map = {'Easy': 1, 'Medium': 2, 'Hard': 3}
            filters = {
                'Cuisine_Type': selected_cuisine,
                'Cooking_Time_Minutes': prep_time,
                'Calories_Per_Serving': calories,
                'Difficulty_Score': dif_score_map[difficulty],
                'Season': season,
                'Occasion': occasion,
                'Popularity_Score': 5.0 # Default fallback
            }
            
            recs, total_candidates, nearest_cluster, fallback_flag = rec_engine.recommend_recipes(filters, top_n=6)
            
            if recs.empty:
                st.warning("Could not find matching recipes. Please run full training first or lift restrictions.")
            else:
                if fallback_flag:
                    st.warning("⚠️ No recipes perfectly matched all strict slider bounds! Showing fallback recommendations with the same **Cuisine**, **Season**, and **Occasion** to fit your core request.")
                else:
                    st.success(f"Top matches for your {selected_cuisine} craving found!")
                
                # Compare Outputs Explanation based on current run
                st.info(f"""
                🧠 **Algorithm Output Comparison**:
                - **Clustering**: Grouped your filter properties and narrowed full recipes down to **Cluster {nearest_cluster}** containing **{total_candidates}** similar candidates.
                - **Classification**: Score-sorted that dense subset using decision criteria. Below are your top **{len(recs)}** high-probability matches!
                """)
                
                # Save latest search to session state for Insights comparison
                st.session_state['latest_filters'] = filters
                st.session_state['recs_loaded'] = True
                
                # Render Cards 3-column Grid
                cols = st.columns(3)
                for index, (_, row) in enumerate(recs.iterrows()):
                    with cols[index % 3]:
                        # Unified Glassmorphic Container
                        with st.container(border=True):
                            st.markdown(f"""
                            <div class="recipe-title">{"👑 Top Pick: " if index == 0 else ""}{row['Recipe_Name']}</div>
                            <div>
                                <span class="badge" style="background: rgba(255, 127, 80, 0.15); color: #ff7f50;">{row['Cuisine_Type']}</span>
                                <span class="badge" style="color: #4dd0e1;">{row['Difficulty_Level']}</span>
                            </div>
                            <div style="margin-top: 15px; margin-bottom: 20px; display: flex; justify-content: space-between;">
                                <div>
                                    <div class="metric-label">Prep Time</div>
                                    <div class="metric-value">{row['Cooking_Time_Minutes']} m</div>
                                </div>
                                <div>
                                    <div class="metric-label">Calories</div>
                                    <div class="metric-value">{row['Calories_Per_Serving']} kcal</div>
                                </div>
                                <div>
                                    <div class="metric-label">Liking Match</div>
                                    <div class="metric-value" style="color: {'#32cd32' if row.get('Liking_Probability', 0.5) > 0.7 else '#ff8c00' if row.get('Liking_Probability', 0.5) > 0.4 else '#ff4500'};">{row.get('Liking_Probability', 0.5)*100:.1f}%</div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Toggle Details inside Session State
                            expand_key = f"expand_{row['Recipe_ID']}_{index}"
                            
                            # Standard Button placed cleanly inside container footer
                            if st.button("🔍 View Details & Ratings", key=f"btn_{row['Recipe_ID']}_{index}", use_container_width=True):
                                st.session_state[expand_key] = not st.session_state.get(expand_key, False)
                                
                            # Inside Detail View Section (Inline)
                            if st.session_state.get(expand_key, False):
                                st.markdown(f"""
                                <div style="background: rgba(255,255,255,0.02); padding: 15px; border-top: 1px dashed rgba(255,127,80,0.3); border-radius: 8px; margin-top: 10px;">
                                    <p style="color: #ff7f50; font-weight: 600; margin-bottom: 5px;">⭐ Popularity: {row['Popularity_Score']}/10</p>
                                    <p style="color: #aaaaaa; font-size: 0.9rem;">💰 Cost: ${row['Cost_Per_Serving']} /serving</p>
                                    <hr style="border: 0.5px solid rgba(255,255,255,0.05); margin: 10px 0;">
                                    <p style="font-weight: 600; font-size: 0.95rem; color: #ffd700;">📋 Ingredients:</p>
                                    <ul style="padding-left: 20px; font-size: 0.85rem; color: #dddddd;">
                                        {"".join([f"<li>{i.strip().capitalize()}</li>" for i in str(row['Ingredients_List']).split(",") if i.strip()])}
                                    </ul>
                                    <p style="font-weight: 600; font-size: 0.95rem; color: #4dd0e1; margin-top: 10px;">🍳 Prep Steps:</p>
                                    <div style="font-size: 0.85rem; color: #dddddd; line-height: 1.4;">
                                        {"".join([f"<p style='margin-bottom: 4px;'><b>{idx+1}.</b> {s.strip().capitalize()}</p>" for idx, s in enumerate(str(row['Preparation_Steps']).split(".")) if s.strip() and not s.strip().isdigit()])}
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
    else:
        st.info("👈 Adjust filters in the sidebar and click 'Recommend Recipes' to start!")

with tab2:
    st.markdown("### 📊 Recipe Inventory & Cluster Analytics")
    if rec_engine.df is not None:
        df_unique = rec_engine.df.drop_duplicates(subset=['Recipe_ID'])
        
        # Dashboard Overview Section
        m1, m2, m3 = st.columns(3)
        with m1:
             st.markdown(f'<div class="dash-metric"><h3>Total Recipes</h3><p style="font-size: 1.5rem; font-weight: 700;">{len(df_unique)}</p></div>', unsafe_allow_html=True)
        with m2:
             st.markdown(f'<div class="dash-metric"><h3>Avg Calories</h3><p style="font-size: 1.5rem; font-weight: 700;">{int(df_unique["Calories_Per_Serving"].mean())} kcal</p></div>', unsafe_allow_html=True)
        with m3:
             st.markdown(f'<div class="dash-metric"><h3>Cuisine Variety</h3><p style="font-size: 1.5rem; font-weight: 700;">{df_unique["Cuisine_Type"].nunique()}</p></div>', unsafe_allow_html=True)
             
        st.markdown("---")
        
        # 0. User Current Cluster Match
        if 'latest_filters' in st.session_state:
            input_data = pd.DataFrame([st.session_state['latest_filters']])
            preprocessor = rec_engine.clusterer_pipe.named_steps['preprocessor']
            X_clust = preprocessor.transform(input_data)
            current_cluster = rec_engine.clusterer_pipe.named_steps['clusterer'].predict(X_clust)[0]
            
            st.markdown(f"#### 🎯 Current Preference Location")
            st.info(f"Based on your sidebar filters, your preferences fall into **Cluster {current_cluster}**.")
            st.markdown("---")

        # 1. Cuisine Distribution Plotly
        st.markdown("#### 🌍 Recipe Distribution by Cuisine")
        cuisine_counts = df_unique['Cuisine_Type'].value_counts().reset_index()
        cuisine_counts.columns = ['Cuisine_Type', 'Count']
        fig_cuisine = px.bar(cuisine_counts, x='Cuisine_Type', y='Count', 
                             color='Count', color_continuous_scale='Oranges',
                             labels={'Count': 'Number of Recipes'},
                             template='plotly_dark')
        st.plotly_chart(fig_cuisine, use_container_width=True)
        
        # 2. Clusters Breakdown Plotly
        if 'Cluster' in rec_engine.df.columns:
            st.markdown("#### 🌀 KMeans Clusters Separation")
            fig_cluster = px.scatter(df_unique, x='Cooking_Time_Minutes', y='Calories_Per_Serving',
                                      color='Cluster', size='Popularity_Score', 
                                      hover_name='Recipe_Name', 
                                      color_continuous_scale='Plasma',
                                      labels={'Cooking_Time_Minutes': 'Prep Time (m)', 'Calories_Per_Serving': 'Calories'},
                                      title="Clusters size weighted by Popularity Score",
                                      template='plotly_dark')
            st.plotly_chart(fig_cluster, use_container_width=True)
            
        # 3. Algorithm Comparison Section
        st.markdown("---")
        st.markdown("### 🤝 Clustering vs Classification in Recommendations")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### 🌀 Clustering (KMeans)")
            st.markdown("""
            *   **Grouping**: Unsupervised algorithm finding item structural similarity.
            *   **Usage**: Maps user query profile properties directly into pre-mapped recipe groups.
            *   **Benefit**: Identifies thematic groups like *Quick Italian* vs *Slow Slow cooked desserts*.
            """)
        with c2:
            st.markdown("#### 🌳 Classification (Decision Tree)")
            st.markdown("""
            *   **Ranking**: Supervised algorithm predicting scoring criteria tiers.
            *   **Usage**: Weighs User preferences matching recipe dimensions for deterministic likability thresholds.
            *   **Benefit**: Extracts precise triggers like *Cuisine matching matches* boosting accurate top rankings setup.
            """)
    else:
        st.error("System Data not processed. Run `python src/training_pipeline.py` first.")
