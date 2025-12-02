import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry
import plotly.express as px
import plotly.graph_objects as go
import os
import numpy as np

# Setup Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

def fetch_climate_data(lat, lon, start_year, end_year):
    """Fetches annual mean temperature and total precipitation for a location."""
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": f"{start_year}-01-01",
        "end_date": f"{end_year}-12-31",
        "daily": ["temperature_2m_mean", "precipitation_sum"],
        "timezone": "auto"
    }
    
    try:
        responses = openmeteo.weather_api(url, params=params)
        response = responses[0]
        
        # Process daily data
        daily = response.Daily()
        daily_temperature_2m_mean = daily.Variables(0).ValuesAsNumpy()
        daily_precipitation_sum = daily.Variables(1).ValuesAsNumpy()
        
        daily_data = {"date": pd.date_range(
            start=pd.to_datetime(daily.Time(), unit="s", utc=True),
            end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=daily.Interval()),
            inclusive="left"
        )}
        daily_data["temperature_2m_mean"] = daily_temperature_2m_mean
        daily_data["precipitation_sum"] = daily_precipitation_sum
        
        df_daily = pd.DataFrame(data=daily_data)
        
        # Aggregate to annual mean/sum
        mean_temp = df_daily['temperature_2m_mean'].mean()
        total_precip = df_daily['precipitation_sum'].sum() # Total precip over the period
        
        # Normalize precip to annual average if multiple years
        years = end_year - start_year + 1
        avg_annual_precip = total_precip / years
        
        return mean_temp, avg_annual_precip
        
    except Exception as e:
        print(f"Error fetching data for {lat}, {lon}: {e}")
        return None, None

def main():
    print("Loading cleaned dataset...")
    if not os.path.exists('cleaned_dataset.csv'):
        print("Error: cleaned_dataset.csv not found.")
        return
        
    df = pd.read_csv('cleaned_dataset.csv', low_memory=False)
    
    # Filter for Top 5 Countries and Recent Years (2010-2023)
    print("Filtering data...")
    top_countries = df['countryCode'].value_counts().head(5).index.tolist()
    df_filtered = df[
        (df['countryCode'].isin(top_countries)) & 
        (df['year'] >= 2010) & 
        (df['year'] <= 2023)
    ].copy()
    
    # Group by Country and Year to get Centroids and Species Counts
    print("Aggregating data...")
    agg_data = df_filtered.groupby(['countryCode', 'year']).agg({
        'decimalLatitude': 'mean',
        'decimalLongitude': 'mean',
        'species': 'nunique', # Species Richness
        'gbifID': 'count'     # Observation Count
    }).reset_index()
    
    print(f"Processing {len(agg_data)} aggregated records...")
    
    # Fetch Climate Data for each Country-Year
    temps = []
    precips = []
    
    for index, row in agg_data.iterrows():
        print(f"Fetching climate data for {row['countryCode']} - {row['year']}...")
        # Fetch data for that specific year
        temp, precip = fetch_climate_data(row['decimalLatitude'], row['decimalLongitude'], int(row['year']), int(row['year']))
        temps.append(temp)
        precips.append(precip)
        
    agg_data['mean_temp'] = temps
    agg_data['total_precip'] = precips
    
    # Drop rows where climate data failed
    agg_data = agg_data.dropna(subset=['mean_temp', 'total_precip'])
    
    # Correlation Analysis
    print("Performing correlation analysis...")
    corr_temp = agg_data['species'].corr(agg_data['mean_temp'])
    corr_precip = agg_data['species'].corr(agg_data['total_precip'])
    
    print(f"Correlation (Temp vs Richness): {corr_temp}")
    print(f"Correlation (Precip vs Richness): {corr_precip}")
    
    # Visualizations
    if not os.path.exists('images'):
        os.makedirs('images')
        
    # Temp vs Richness
    fig_temp = px.scatter(
        agg_data, x='mean_temp', y='species', color='countryCode',
        title=f'Temperature vs. Species Richness (Corr: {corr_temp:.2f})',
        labels={'mean_temp': 'Mean Annual Temperature (°C)', 'species': 'Species Richness'}
    )
    fig_temp.write_html("images/climate_correlation_temp.html")
    try:
        fig_temp.write_image("images/climate_correlation_temp.png")
    except:
        pass
        
    # Precip vs Richness
    fig_precip = px.scatter(
        agg_data, x='total_precip', y='species', color='countryCode',
        title=f'Precipitation vs. Species Richness (Corr: {corr_precip:.2f})',
        labels={'total_precip': 'Total Annual Precipitation (mm)', 'species': 'Species Richness'}
    )
    fig_precip.write_html("images/climate_correlation_precip.html")
    try:
        fig_precip.write_image("images/climate_correlation_precip.png")
    except:
        pass
        
    print("Climate integration complete. Images saved to 'images/' directory.")

if __name__ == "__main__":
    main()
