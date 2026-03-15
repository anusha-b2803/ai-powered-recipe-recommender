import streamlit as st
import base64
import os

def set_page_config():
    st.set_page_config(
        page_title="AI Powered Recipe Recommender",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )

def inject_custom_css():
    """
    Injects aesthetic Glassmorphism style CSS with background.jpg image base64 format.
    """
    img_path = r'a:\MP\DA MP\recipe-recommender\assets\background.jpg'
    if os.path.exists(img_path):
        with open(img_path, "rb") as f:
            b64_img = base64.b64encode(f.read()).decode()
        bg_style = f"""
        .stApp {{
            background: linear-gradient(rgba(18, 18, 20, 0.85), rgba(26, 27, 38, 0.85)), url("data:image/jpeg;base64,{b64_img}");
            background-size: cover;
            background-attachment: fixed;
            background-position: center;
            color: #ffffff;
        }}
        """
    else:
        bg_style = """
        .stApp {
            background: linear-gradient(135deg, #121214 0%, #1a1b26 50%, #151a22 100%);
            color: #ffffff;
        }
        """

    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
        
        * {
            font-family: 'Poppins', sans-serif;
        }

        """ + bg_style + """
        
        /* Glassmorphic Sidebar */
        [data-testid="stSidebar"] {
            background-color: rgba(25, 27, 38, 0.7) !important;
            backdrop-filter: blur(10px);
            border-right: 1px solid rgba(255, 255, 255, 0.05);
        }

        /* Glassmorphic Native Containers */
        div[data-testid="stVerticalBlockBorder"] {
            background: rgba(255, 255, 255, 0.03) !important;
            border: 1px solid rgba(255, 255, 255, 0.05) !important;
            border-radius: 16px !important;
            padding: 20px !important;
            backdrop-filter: blur(5px);
            transition: all 0.3s ease;
            box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
            margin-bottom: 20px;
        }
        
        div[data-testid="stVerticalBlockBorder"]:hover {
            transform: translateY(-5px);
            background: rgba(255, 255, 255, 0.06) !important;
            border: 1px solid rgba(255, 107, 107, 0.3) !important;
            box-shadow: 0 10px 40px rgba(255, 107, 107, 0.1);
        }

        .recipe-title {
            color: #ff7f50;
            font-size: 1.25rem;
            font-weight: 600;
            margin-bottom: 10px;
        }
        
        .badge {
            display: inline-block;
            padding: 4px 10px;
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: 600;
            margin-right: 8px;
            margin-bottom: 5px;
            background: rgba(255, 255, 255, 0.1);
            color: #ffd700;
        }
        
        .metric-label {
            color: #aaaaaa;
            font-size: 0.85rem;
        }
        
        .metric-value {
            color: #ffffff;
            font-weight: 600;
            font-size: 1rem;
        }
        
        /* Custom sidebar headers */
        .sidebar-header {
            font-size: 1.5rem;
            font-weight: 700;
            color: #ff7f50;
            margin-bottom: 1rem;
        }
        
        /* Dashboard metric header */
        .dash-metric {
            background: rgba(43, 44, 53, 0.4);
            padding: 15px;
            border-radius: 12px;
            text-align: center;
            border-left: 4px solid #ff7f50;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

def render_recipe_card(name, cuisine, time, difficulty, calories, matching_prob):
    """
    Renders a HTML/CSS grid card using Streamlit markdown expansion.
    """
    prob_color = "#32cd32" if matching_prob > 0.7 else "#ff8c00" if matching_prob > 0.4 else "#ff4500"
    
    st.markdown(
        f"""
        <div class="recipe-card">
            <div class="recipe-title">{name}</div>
            <div>
                <span class="badge" style="background: rgba(255, 127, 80, 0.15); color: #ff7f50;">{cuisine}</span>
                <span class="badge" style="color: #4dd0e1;">{difficulty}</span>
            </div>
            <div style="margin-top: 15px; display: flex; justify-content: space-between;">
                <div>
                    <div class="metric-label">Prep Time</div>
                    <div class="metric-value">{time} m</div>
                </div>
                <div>
                    <div class="metric-label">Calories</div>
                    <div class="metric-value">{calories} kcal</div>
                </div>
                <div>
                    <div class="metric-label">Liking Match</div>
                    <div class="metric-value" style="color: {prob_color};">{matching_prob*100:.1f}%</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
