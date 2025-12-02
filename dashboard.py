import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
import openmeteo_requests
import requests_cache
from retry_requests import retry

# ---------------------------------------------------------
# Page Configuration & Custom CSS
# ---------------------------------------------------------
st.set_page_config(
    page_title="Bio-Explorer: Global Nature Tracker",
    page_icon="🦁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Youth-Friendly Theme
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;800&display=swap');

        /* Main Background */
        .stApp {
            background: linear-gradient(135deg, #E0F7FA 0%, #E8F5E9 100%);
            font-family: 'Poppins', sans-serif;
        }
        
        /* Sidebar */
        [data-testid="stSidebar"] {
            background-color: #FFFFFF;
            border-right: 2px solid #81C784;
        }
        
        /* Headers */
        h1 {
            color: #2E7D32;
            font-weight: 800;
            text-shadow: 2px 2px 0px #A5D6A7;
        }
        h2, h3 {
            color: #00695C;
            font-weight: 600;
        }
        
        /* Cards */
        .metric-card {
            background-color: white;
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            text-align: center;
            border: 2px solid #B2DFDB;
        }
        
        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 10px;
            background-color: transparent;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            background-color: #FFFFFF;
            border-radius: 25px;
            padding: 0px 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            border: 1px solid #E0E0E0;
            font-weight: 600;
        }
        .stTabs [aria-selected="true"] {
            background-color: #FFEB3B; /* Sunny Yellow */
            color: #2E7D32;
            border: 2px solid #FBC02D;
        }
        
        /* Buttons */
        .stButton button {
            background-color: #FF7043; /* Vibrant Orange */
            color: white;
            border-radius: 25px;
            font-weight: bold;
            border: none;
            box-shadow: 0 4px 0 #D84315;
            transition: all 0.1s;
        }
        .stButton button:active {
            box-shadow: 0 0 0 #D84315;
            transform: translateY(4px);
        }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Data Loading
# ---------------------------------------------------------
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('cleaned_dataset.csv', low_memory=False)
        return df
    except FileNotFoundError:
        st.error("cleaned_dataset.csv not found. Please run data cleaning first.")
        return pd.DataFrame()

df = load_data()

# ---------------------------------------------------------
# Sidebar Filters
# ---------------------------------------------------------
st.sidebar.image("https://img.icons8.com/color/96/000000/sloth.png", width=80)
st.sidebar.title("🌍 Explorer's Kit")
st.sidebar.markdown("**Filter your adventure!**")

if not df.empty:
    # Country Filter
    countries = sorted(df['countryCode'].dropna().unique())
    if st.sidebar.checkbox("Select All Countries"):
        selected_countries = countries
    else:
        selected_countries = st.sidebar.multiselect("Select Countries", countries, default=countries[:5] if len(countries) > 5 else countries)
    
    # Year Filter
    min_year = int(df['year'].min())
    max_year = int(df['year'].max())
    selected_years = st.sidebar.slider("Time Travel (Year)", min_year, max_year, (2010, max_year))
    
    # Kingdom Filter
    kingdoms = sorted(df['kingdom'].unique())
    if st.sidebar.checkbox("Select All Kingdoms", value=True):
        selected_kingdoms = kingdoms
    else:
        selected_kingdoms = st.sidebar.multiselect("Select Kingdoms", kingdoms, default=kingdoms)

    # Apply Filters
    filtered_df = df[
        (df['countryCode'].isin(selected_countries)) &
        (df['year'] >= selected_years[0]) &
        (df['year'] <= selected_years[1]) &
        (df['kingdom'].isin(selected_kingdoms))
    ]
else:
    filtered_df = pd.DataFrame()

# ---------------------------------------------------------
# Main Dashboard
# ---------------------------------------------------------
st.title("🍃 Bio-Explorer: Global Nature Tracker")
st.markdown("### Welcome, Young Scientist!")
st.markdown("Dive into the amazing world of biodiversity. Discover where animals and plants live, how the climate affects them, and become a data detective!")

if filtered_df.empty:
    st.warning("Uh oh! No creatures found with these filters. Try changing your Explorer's Kit settings!")
else:
    # Tabs
    # Tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["🌍 Global Snapshot", "📊 Fun Facts & Stats", "🧠 Deep Dive", "☀️ Climate & Life", "📍 Find a Creature", "🇮🇳 India Special"])

    # ---------------------------------------------------------
    # Tab 1: Overview
    # ---------------------------------------------------------
    with tab1:
        st.markdown("#### What's happening out there?")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Sightings", f"{len(filtered_df):,}", delta="Observed")
        col2.metric("Unique Species", f"{filtered_df['species'].nunique():,}", delta="Discovered")
        col3.metric("Countries", f"{filtered_df['countryCode'].nunique()}", delta="Explored")
        
        st.subheader("📍 Recent Observations Map")
        st.caption("Hover over points to see details!")
        # Sample for map performance
        map_data = filtered_df.head(1000)
        fig_map = px.scatter_mapbox(map_data, lat="decimalLatitude", lon="decimalLongitude",
                                    hover_name="scientificName",
                                    hover_data=["kingdom", "countryCode", "year"],
                                    color_discrete_sequence=["#2E7D32"],
                                    zoom=1, height=500)
        fig_map.update_layout(mapbox_style="open-street-map")
        fig_map.update_traces(marker=dict(size=6))
        st.plotly_chart(fig_map, use_container_width=True)

    # ---------------------------------------------------------
    # Tab 2: Exploratory Analysis
    # ---------------------------------------------------------
    with tab2:
        st.markdown("#### Let's look at the numbers!")
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("👑 Kingdom Battle")
            st.caption("Which kingdom has the most sightings? Plants? Animals?")
            kingdom_counts = filtered_df['kingdom'].value_counts().reset_index()
            kingdom_counts.columns = ['Kingdom', 'Count']
            fig_kingdom = px.bar(kingdom_counts, x='Kingdom', y='Count', color='Kingdom', 
                                 color_discrete_sequence=px.colors.qualitative.Bold, title="Sightings by Kingdom")
            st.plotly_chart(fig_kingdom, use_container_width=True)
            
        with col2:
            st.subheader("🏆 Top Countries")
            st.caption("Which countries are the biodiversity hotspots?")
            country_counts = filtered_df['countryCode'].value_counts().head(10).reset_index()
            country_counts.columns = ['Country', 'Count']
            fig_country = px.bar(country_counts, x='Country', y='Count', color='Count', 
                                 color_continuous_scale='Viridis', title="Top 10 Countries")
            st.plotly_chart(fig_country, use_container_width=True)
            
        st.subheader("📅 Timeline of Discovery")
        st.caption("Are we finding more creatures over time?")
        yearly_counts = filtered_df['year'].value_counts().sort_index().reset_index()
        yearly_counts.columns = ['Year', 'Count']
        fig_trend = px.area(yearly_counts, x='Year', y='Count', markers=True, 
                            color_discrete_sequence=['#FF7043'], title="Sightings Over Time")
        st.plotly_chart(fig_trend, use_container_width=True)

    # ---------------------------------------------------------
    # Tab 3: Advanced Extensions
    # ---------------------------------------------------------
    with tab3:
        st.markdown("#### 🧠 Become a Data Detective")
        
        with st.expander("🧬 What is a 'Biodiversity Score'?"):
            st.write("Scientists use a special math formula called the **Shannon Index** to measure how diverse an area is. A higher score means there are many different types of species living together happily! It's like a 'Health Score' for nature.")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Biodiversity Score")
            if st.button("Calculate Scores"):
                with st.spinner("Crunching the numbers..."):
                    # Shannon Index Logic
                    def calculate_shannon(data):
                        top_c = data['countryCode'].value_counts().head(5).index.tolist()
                        results = {}
                        for c in top_c:
                            c_data = data[data['countryCode'] == c]
                            counts = c_data['scientificName'].value_counts()
                            total = counts.sum()
                            props = counts / total
                            results[c] = -np.sum(props * np.log(props))
                        return results

                    shannon_res = calculate_shannon(filtered_df)
                    df_shannon = pd.DataFrame(list(shannon_res.items()), columns=['Country', 'Score'])
                    fig_shannon = px.bar(df_shannon, x='Country', y='Score', color='Score', 
                                         color_continuous_scale='Plasma', title="Biodiversity Health Score")
                    st.plotly_chart(fig_shannon, use_container_width=True)

        with col2:
            st.subheader("🔮 Future Predictions")
            st.caption("Can AI guess how many creatures we'll find in 2030?")
            if st.button("Ask Me"):
                with st.spinner("Gazing into the crystal ball..."):
                    yearly = filtered_df['year'].value_counts().sort_index().reset_index()
                    yearly.columns = ['year', 'count']
                    model_data = yearly[yearly['year'] >= 1950]
                    
                    if len(model_data) > 1:
                        X = model_data[['year']]
                        y = model_data['count']
                        model = LinearRegression()
                        model.fit(X, y)
                        
                        future_years = np.array([[2024], [2025], [2030]])
                        predictions = model.predict(future_years)
                        
                        fig_pred = go.Figure()
                        fig_pred.add_trace(go.Scatter(x=model_data['year'], y=model_data['count'], mode='markers', name='Real Data'))
                        fig_pred.add_trace(go.Scatter(x=model_data['year'], y=model.predict(X), mode='lines', name='Trend Line', line=dict(color='orange', dash='dash')))
                        fig_pred.add_trace(go.Scatter(x=future_years.flatten(), y=predictions, mode='markers+text', name='AI Prediction', 
                                                      text=[f"{int(p)}" for p in predictions], textposition="top center",
                                                      marker=dict(color='purple', size=12, symbol='star')))
                        fig_pred.update_layout(title="Future Sighting Forecast")
                        st.plotly_chart(fig_pred, use_container_width=True)
                    else:
                        st.warning("Not enough data to predict the future yet!")

        st.markdown("---")
        st.subheader("🗺️ Secret Hotspots (Clustering)")
        st.caption("Where are the biggest gatherings of creatures?")
        if st.button("Reveal Hotspots"):
            with st.spinner("Scanning satellite data..."):
                coords = filtered_df[['decimalLatitude', 'decimalLongitude']].dropna()
                if len(coords) > 10000:
                    coords = coords.sample(10000, random_state=42)
                
                kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
                coords['cluster'] = kmeans.fit_predict(coords)
                
                fig_clusters = px.scatter_mapbox(coords, lat="decimalLatitude", lon="decimalLongitude", color="cluster",
                                        zoom=1, height=500, title='5 Major Biodiversity Hotspots')
                fig_clusters.update_layout(mapbox_style="open-street-map")
                st.plotly_chart(fig_clusters, use_container_width=True)

    # ---------------------------------------------------------
    # Tab 4: Climate Correlations
    # ---------------------------------------------------------
    with tab4:
        st.markdown("#### ☀️ Sun, Rain, and Life")
        st.info("Does the weather affect where animals live? Let's find out using real climate data!")
        
        if st.button("Analyze Climate Data"):
            with st.spinner("Connecting to weather satellites..."):
                # Simplified logic for dashboard
                top_countries_list = filtered_df['countryCode'].value_counts().head(5).index.tolist()
                subset_df = filtered_df[filtered_df['countryCode'].isin(top_countries_list)].copy()
                
                agg_data = subset_df.groupby(['countryCode', 'year']).agg({
                    'decimalLatitude': 'mean',
                    'decimalLongitude': 'mean',
                    'species': 'nunique'
                }).reset_index()
                
                # Setup API
                cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
                retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
                openmeteo = openmeteo_requests.Client(session=retry_session)
                
                temps = []
                precips = []
                
                progress_bar = st.progress(0)
                total_rows = len(agg_data)
                
                for i, row in agg_data.iterrows():
                    try:
                        url = "https://archive-api.open-meteo.com/v1/archive"
                        params = {
                            "latitude": row['decimalLatitude'],
                            "longitude": row['decimalLongitude'],
                            "start_date": f"{int(row['year'])}-01-01",
                            "end_date": f"{int(row['year'])}-12-31",
                            "daily": ["temperature_2m_mean", "precipitation_sum"],
                            "timezone": "auto"
                        }
                        responses = openmeteo.weather_api(url, params=params)
                        response = responses[0]
                        daily = response.Daily()
                        t = daily.Variables(0).ValuesAsNumpy().mean()
                        p = daily.Variables(1).ValuesAsNumpy().sum() / (1 if (daily.TimeEnd() - daily.Time()) < 31536000 else 1) # Approx annual
                        temps.append(t)
                        precips.append(p)
                    except Exception as e:
                        temps.append(None)
                        precips.append(None)
                    progress_bar.progress((i + 1) / total_rows)
                
                agg_data['mean_temp'] = temps
                agg_data['total_precip'] = precips
                agg_data = agg_data.dropna()
                
                if not agg_data.empty:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        corr_temp = agg_data['species'].corr(agg_data['mean_temp'])
                        st.markdown(f"**Temperature Connection:** {corr_temp:.2f}")
                        fig_temp = px.scatter(agg_data, x='mean_temp', y='species', color='countryCode',
                                            title='Temperature vs. Species Variety', 
                                            labels={'mean_temp': 'Avg Temperature (°C)', 'species': 'Number of Species'})
                        st.plotly_chart(fig_temp, use_container_width=True)
                        
                    with col2:
                        corr_precip = agg_data['species'].corr(agg_data['total_precip'])
                        st.markdown(f"**Rainfall Connection:** {corr_precip:.2f}")
                        fig_precip = px.scatter(agg_data, x='total_precip', y='species', color='countryCode',
                                            title='Rainfall vs. Species Variety', 
                                            labels={'total_precip': 'Total Rainfall (mm)', 'species': 'Number of Species'})
                        st.plotly_chart(fig_precip, use_container_width=True)
                else:
                    st.error("Could not get weather data. Try again later!")

    # ---------------------------------------------------------
    # Tab 5: Species Map (New)
    # ---------------------------------------------------------
    with tab5:
        st.markdown("#### 📍 Find a Creature")
        st.info("Pick a species to see exactly where it has been spotted!")
        
        # Species Search Box
        all_species = sorted(filtered_df['scientificName'].dropna().unique())
        if all_species:
            selected_species = st.selectbox("🔍 Type a name (e.g., 'Panthera leo')", all_species)
            
            if selected_species:
                species_data = filtered_df[filtered_df['scientificName'] == selected_species]
                st.success(f"Wow! We found **{len(species_data)}** sightings for *{selected_species}*.")
                
                if not species_data.empty:
                    # Create Map
                    avg_lat = species_data['decimalLatitude'].mean()
                    avg_lon = species_data['decimalLongitude'].mean()
                    m = folium.Map(location=[avg_lat, avg_lon], zoom_start=4)
                    
                    # Add Markers (using Cluster for performance)
                    from folium.plugins import MarkerCluster
                    marker_cluster = MarkerCluster().add_to(m)
                    
                    for idx, row in species_data.iterrows():
                        folium.Marker(
                            location=[row['decimalLatitude'], row['decimalLongitude']],
                            popup=f"{row['scientificName']} ({row['year']})",
                            tooltip=f"{row['scientificName']} - {row['year']}",
                            icon=folium.Icon(color="green", icon="leaf")
                        ).add_to(marker_cluster)
                    
                    st_folium(m, width=700, height=500)
                else:
                    st.warning("No valid coordinates found for this species.")
        else:
            st.warning("No species found with current filters.")

    # ---------------------------------------------------------
    # Tab 6: India Special
    # ---------------------------------------------------------
    with tab6:
        st.markdown("#### 🇮🇳 Namaste! Welcome to the India Special")
        st.info("Exploring the incredible biodiversity of India.")
        
        # Filter for India
        india_df = df[df['countryCode'] == 'IN'].copy()
        
        if not india_df.empty:
            col1, col2 = st.columns(2)
            col1.metric("Total Indian Sightings", f"{len(india_df):,}")
            col2.metric("Unique Species in India", f"{india_df['species'].nunique():,}")
            
            st.subheader("🗺️ Biodiversity Map of India")
            st.caption("Explore where different species have been spotted across India!")
            
            # Static Map (Interactive)
            india_map_data = india_df.sort_values('year')
            fig_anim = px.scatter_mapbox(india_map_data, lat="decimalLatitude", lon="decimalLongitude",
                                         hover_name="scientificName", hover_data=["kingdom", "year"],
                                         color="kingdom", zoom=3, height=600,
                                         title="Biodiversity Sightings in India")
            fig_anim.update_layout(mapbox_style="open-street-map")
            st.plotly_chart(fig_anim, use_container_width=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🦁 Kingdom Distribution")
                fig_pie = px.pie(india_df, names='kingdom', title='Kingdoms of India', 
                                 color_discrete_sequence=px.colors.qualitative.Set3)
                st.plotly_chart(fig_pie, use_container_width=True)
                
            with col2:
                st.subheader("📈 Discovery Trend")
                india_trend = india_df['year'].value_counts().sort_index().reset_index()
                india_trend.columns = ['Year', 'Count']
                fig_line = px.line(india_trend, x='Year', y='Count', markers=True, 
                                   title='Yearly Sightings in India', line_shape='spline',
                                   color_discrete_sequence=['#FF9933']) # Saffron color
                st.plotly_chart(fig_line, use_container_width=True)
                
            st.subheader("🏆 Top 10 Indian Species")
            top_in_species = india_df['scientificName'].value_counts().head(10).reset_index()
            top_in_species.columns = ['Species', 'Count']
            fig_bar_in = px.bar(top_in_species, x='Count', y='Species', orientation='h',
                                title='Most Spotted Species in India', color='Count',
                                color_continuous_scale='Oranges')
            st.plotly_chart(fig_bar_in, use_container_width=True)

            st.subheader("📋 State-wise Biodiversity Data")
            st.caption("Detailed counts of sightings and species across Indian States.")
            
            # State-wise aggregation
            if 'stateProvince' in india_df.columns:
                state_stats = india_df.groupby('stateProvince').agg(
                    Total_Sightings=('scientificName', 'count'),
                    Unique_Species=('scientificName', 'nunique')
                ).reset_index().sort_values('Total_Sightings', ascending=False)
                
                state_stats.columns = ['State/Province', 'Total Sightings', 'Unique Species']
                st.dataframe(state_stats, use_container_width=True)
            else:
                st.warning("State information not available in the dataset.")
            
        else:
            st.warning("No data found for India in the current dataset. Try checking the raw data or updating the dataset!")

    # ---------------------------------------------------------
    # Raw Data Section
    # ---------------------------------------------------------
    st.markdown("---")
    with st.expander("📂 Peek into the Data Vault (Raw Data)"):
        st.write("Here is the data we are using for our analysis:")
        st.dataframe(filtered_df)
        csv = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Data as CSV",
            data=csv,
            file_name='biodiversity_data.csv',
            mime='text/csv',
        )

# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.markdown("""
    <hr>
    <div style='text-align: center; color: #666; padding: 10px;'>
        <p><b>Created by Amit D. Dhumal</b></p>
    </div>
""", unsafe_allow_html=True)

