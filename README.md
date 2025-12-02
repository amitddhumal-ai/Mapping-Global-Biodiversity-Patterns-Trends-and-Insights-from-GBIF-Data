# 🦁 Global Biodiversity Analysis & Bio-Explorer Dashboard

**Created by Amit D. Dhumal**

## 📖 Overview
This project is a comprehensive data science initiative that analyzes global biodiversity patterns using data from the **Global Biodiversity Information Facility (GBIF)**. It goes beyond static analysis by integrating **Climate Data (Open-Meteo)** and presenting insights through an interactive, youth-friendly **Streamlit Dashboard**.

## ✨ Key Features

### 1. 📊 Advanced Data Analysis
-   **Exploratory Data Analysis (EDA)**: Insights into Kingdom distribution, top biodiversity hotspots, and temporal trends.
-   **Scientific Metrics**: Calculation of the **Shannon Diversity Index** to measure ecosystem health.
-   **Machine Learning**:
    -   **K-Means Clustering**: Identifying geographical hotspots of biodiversity.
    -   **Random Forest Regression**: Forecasting future trends in biodiversity data collection with high accuracy.

### 2. ☀️ Climate Integration
-   Integration with the **Open-Meteo Historical Weather API**.
-   Correlation analysis between **Temperature/Precipitation** and **Species Richness**.

### 3. 🍃 Bio-Explorer Dashboard
An interactive web application designed for a younger audience ("Young Scientists"):
-   **🌍 Global Snapshot**: Interactive maps showing recent sightings with hover details.
-   **🧠 Deep Dive**: AI-powered tools to calculate biodiversity scores and predict future trends.
-   **📍 Find a Creature**: Search for specific species (e.g., *Panthera leo*) and view their locations on a map.
-   **🇮🇳 India Special**: A dedicated tab featuring:
    -   Interactive Map of Indian biodiversity.
    -   State-wise data tables.
    -   Kingdom distribution and top species in India.

### 4. 📄 Reports & Presentations
-   **LaTeX Report**: A professional PDF report documenting the entire study.

## 🛠️ Installation

1.  **Clone the repository**:
    ```bash
    git clone <repository-url>
    cd <repository-folder>
    ```

2.  **Install Dependencies**:
    You will need Python installed. Install the required libraries using pip:
    ```bash
    pip install streamlit pandas plotly folium streamlit-folium numpy scikit-learn openmeteo-requests requests-cache retry-requests
    ```

## 🚀 Usage

### Running the Dashboard
To launch the interactive dashboard, run:
```bash
streamlit run dashboard.py
```

### Generating Artifacts
-   **Advanced Analysis Images**:
    ```bash
    python run_advanced_extensions.py
    ```
-   **Climate Analysis**:
    ```bash
    python integrate_climate_data.py
    ```
    ```

## 📂 Project Structure
-   `dashboard.py`: Main Streamlit application.
-   `cleaned_dataset.csv`: Processed biodiversity dataset.
-   `integrate_climate_data.py`: Script for fetching and analyzing climate data.
-   `report.tex`: Source file for the project report (LaTeX).
-   `images/`: Directory containing generated plots and charts.

## 📜 License
This project is for educational and research purposes. Data courtesy of GBIF and Open-Meteo.
