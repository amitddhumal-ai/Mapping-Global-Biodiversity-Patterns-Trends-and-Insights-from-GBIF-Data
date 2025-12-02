import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import os

# Ensure images directory exists
if not os.path.exists('images'):
    os.makedirs('images')

def calculate_shannon_index(df):
    """Calculates Shannon Index for top 5 countries."""
    print("Calculating Shannon Index...")
    top_countries = df['countryCode'].value_counts().head(5).index.tolist()
    shannon_results = {}

    for country in top_countries:
        country_data = df[df['countryCode'] == country]
        species_counts = country_data['scientificName'].value_counts()
        total_individuals = species_counts.sum()
        proportions = species_counts / total_individuals
        shannon_index = -np.sum(proportions * np.log(proportions))
        shannon_results[country] = shannon_index

    return shannon_results

def plot_shannon_index(shannon_results):
    """Plots Shannon Index using Plotly."""
    print("Plotting Shannon Index...")
    df_shannon = pd.DataFrame(list(shannon_results.items()), columns=['Country', 'Shannon Index'])
    fig = px.bar(df_shannon, x='Country', y='Shannon Index', title='Species Richness (Shannon Index) for Top 5 Countries', color='Shannon Index', color_continuous_scale='Magma')
    
    # Save
    # fig.write_html("images/shannon_index.html")
    try:
        fig.write_image("images/shannon_index.png")
    except Exception as e:
        print(f"Error saving PNG: {e}")
    return fig

def perform_cluster_analysis(df):
    """Performs K-Means clustering on geographical coordinates."""
    print("Performing Cluster Analysis...")
    coords = df[['decimalLatitude', 'decimalLongitude']].dropna()

    # Sample if too large
    if len(coords) > 50000:
        coords_sample = coords.sample(50000, random_state=42)
    else:
        coords_sample = coords.copy()

    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    coords_sample['cluster'] = kmeans.fit_predict(coords_sample)

    return coords_sample, kmeans.cluster_centers_

def plot_clusters(coords_sample):
    """Plots geographical clusters using Plotly."""
    print("Plotting Clusters...")
    fig = px.scatter_mapbox(coords_sample, lat="decimalLatitude", lon="decimalLongitude", color="cluster",
                            zoom=1, height=600, title='Geographical Clusters of Observations (K-Means, k=5)')
    fig.update_layout(mapbox_style="open-street-map")
    
    # Save
    # fig.write_html("images/geographical_clusters.html")
    try:
        fig.write_image("images/geographical_clusters.png")
    except Exception as e:
        print(f"Error saving PNG: {e}")
    return fig

def predict_trends(df):
    """Predicts future observation trends using Random Forest Regression."""
    print("Predicting Trends...")
    yearly_counts = df['year'].value_counts().sort_index().reset_index()
    yearly_counts.columns = ['year', 'count']

    # Filter for recent relevant history (e.g., post-1950)
    model_data = yearly_counts[yearly_counts['year'] >= 1950]

    X = model_data[['year']]
    y = model_data['count']

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    future_years = np.array([[2024], [2025], [2030]])
    predictions = model.predict(future_years)

    return model_data, future_years, predictions, model

def plot_predictions(model_data, future_years, predictions, model):
    """Plots trend predictions using Plotly."""
    print("Plotting Predictions...")
    X = model_data[['year']]

    fig = go.Figure()

    # Actual Data
    fig.add_trace(go.Scatter(x=model_data['year'], y=model_data['count'], mode='markers', name='Actual Data'))

    # Trend Line
    fig.add_trace(go.Scatter(x=model_data['year'], y=model.predict(X), mode='lines', name='Trend Line', line=dict(color='red')))

    # Predictions
    fig.add_trace(go.Scatter(x=future_years.flatten(), y=predictions, mode='markers', name='Predictions', marker=dict(color='green', size=10, symbol='x')))

    fig.update_layout(title='Observation Trend and Forecast (Random Forest)', xaxis_title='Year', yaxis_title='Count')
    
    # Save
    # fig.write_html("images/observation_trend_forecast.html")
    try:
        fig.write_image("images/observation_trend_forecast.png")
    except Exception as e:
        print(f"Error saving PNG: {e}")
    return fig

def main():
    input_file = 'cleaned_dataset.csv'
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found. Please run the data cleaning step first.")
        return

    print(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file, low_memory=False)
    
    # Shannon Index
    shannon_results = calculate_shannon_index(df)
    plot_shannon_index(shannon_results)
    
    # Cluster Analysis
    coords_sample, _ = perform_cluster_analysis(df)
    plot_clusters(coords_sample)
    
    # Trend Prediction
    model_data, future_years, predictions, model = predict_trends(df)
    plot_predictions(model_data, future_years, predictions, model)
    
    print("Advanced extensions execution complete. Images saved to 'images/' directory.")

if __name__ == "__main__":
    main()
