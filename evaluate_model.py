import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline

def evaluate_model():
    print("Loading cleaned_dataset.csv...")
    try:
        df = pd.read_csv('cleaned_dataset.csv', low_memory=False)
    except FileNotFoundError:
        print("Error: cleaned_dataset.csv not found.")
        return

    print("Preparing data for Modeling...")
    # Logic matching the notebook
    yearly_counts = df['year'].value_counts().sort_index().reset_index()
    yearly_counts.columns = ['year', 'count']
    
    # Filter for recent relevant history (post-1950)
    model_data = yearly_counts[yearly_counts['year'] >= 1950]
    
    X = model_data[['year']]
    y = model_data['count']
    
    print(f"Training models on {len(model_data)} data points (1950-2023)...")
    
    models = {
        "Linear Regression": LinearRegression(),
        "Polynomial Regression (Degree 2)": make_pipeline(PolynomialFeatures(degree=2), LinearRegression()),
        "Polynomial Regression (Degree 3)": make_pipeline(PolynomialFeatures(degree=3), LinearRegression()),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42)
    }
    
    results = {}
    
    print("\nModel Evaluation Results:")
    print("-" * 60)
    print(f"{'Model':<35} | {'R2':<10} | {'MAE':<10} | {'RMSE':<10}")
    print("-" * 60)
    
    for name, model in models.items():
        model.fit(X, y)
        y_pred = model.predict(X)
        
        r2 = r2_score(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        
        results[name] = {'R2': r2, 'MAE': mae, 'RMSE': rmse}
        
        print(f"{name:<35} | {r2:.4f}     | {mae:.2f}      | {rmse:.2f}")

    print("-" * 60)
    
    # Find best model based on R2
    best_model_name = max(results, key=lambda x: results[x]['R2'])
    print(f"\nBest Performing Model: {best_model_name}")

if __name__ == "__main__":
    evaluate_model()
