# AI Powered Recipe Recommender

**[Live Demo on Render](https://YOUR_APP_NAME.onrender.com)**

An industry-grade recommendation engine combining **KMeans Clustering** and **Decision Tree Classification** with a sleek interactive **Streamlit Dashboard**.

## Project Architecture

- **Data Engineering**: Synthetic data generator producing 10k correlated user-recipe rows.
- **Preprocessing Pipeline**: Missing value imputation, target conversion, and Standard Scaling wrappers.
- **Feature Engineering**: Calculates cook bins, difficulty maps, ingredient length, and cosine-match scores.
- **Clustering**: Groups recipes based on features (Cuisine, Calories, Time) to optimize search density.
- **Classification**: Regularized CART Decision Tree tuned using `GridSearchCV` estimating true liking probabilities.
- **Interface**: Glassmorphic dark-mode dashboard showcasing cards layout structure with high-refresh viz widgets.

---

## Setup Instructions

### 1. Install Dependencies
Make sure you are in the project folder root:
```bash
pip install -r requirements.txt
```

### 2. Run Data Generator & Training
Generate raw assets setting and fit both Cluster and Classification metrics models iteratively into persistence bins.
```bash
python src/data_generator.py
$env:PYTHONPATH="."; python src/training_pipeline.py
```

### 3. Launch Streamlit UI
Run Streamlit directly targeted linking components tree:
```bash
streamlit run app/streamlit_app.py
```

## Evaluation Standards
- **Clustering**: Evaluated with Silhouette Scores.
- **Classification**: Verified over `Accuracy`, `Recall`, and `F1 Score` preventing overfit defaults weights.
- **Transparency**: Code aligns cleanly using `Scikit-Learn Pipeline Pattern` guidelines.

---

## Deployment on Render

To deploy this Streamlit app to **Render.com**:

1.  **Create a New Web Service** on Render connected to your GitHub repository dashboard.
2.  Set **Environment** setup configuration type to **`Python`**.
3.  Set **Branch** bounds setting to `main` or your active branch.
4.  **Build Command**:
    ```bash
    pip install -r requirements.txt
    ```
5.  **Start Command**:
    ```bash
    streamlit run app/streamlit_app.py --server.port $PORT
    ```
6.  Click **Create Web Service** at footer bounds dashboard and wait for completion. Use the resulting address link with this document above for referencing!
