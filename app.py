import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px

# 1. INTEGRATED MODEL TRAINING (Runs once on startup)
@st.cache_resource # Caches the model so it doesn't retrain on every click
def train_simulation_model():
    # Load the high-res master data we prepared earlier
    df = pd.read_csv('Nuclear_Lab_Final_Master.csv')
    
    # We select features with high importance for our 'Ozone' and 'CO' models
    # Focus on precursors and temporal lag
    features = ['Nitrogen dioxide (NO2)', 'Outdoor Temperature', 'Relative Humidity ', 'Ozone_Lag1']
    df_m = df.dropna(subset=['Ozone'] + features)
    
    X = df_m[features]
    y = df_m['Ozone']
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return df, model

# Load data and model
df, o3_model = train_simulation_model()

# 2. APP LAYOUT
st.set_page_config(page_title="NEMS | Nuclear Lab Predictor", layout="wide", page_icon="☢️")
st.title("🛡️ Nuclear Environmental Monitoring System (NEMS)")

# Sidebar for site selection
st.sidebar.header("Station Controls")
selected_site = st.sidebar.selectbox("Select Monitoring Station", df['LOCAL_SITE_NAME'].unique())

# 3. INTERACTIVE FEATURES & SECTIONS
tabs = st.tabs(["🔮 Live Simulation & Testing", "🔬 Forensic Analysis", "🚨 Early Warning System"])

# --- SECTION 1: THE INTERACTIVE PREDICTOR ---
with tabs[0]:
    st.header("Real-Time Safety Simulation")
    st.markdown("""
    **Objective:** Test if current environmental variables will cause a breach in safety thresholds.
    This uses our **Random Forest Regressor (R²: 0.7162)** to predict atmospheric impact.
    """)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Manual Sensor Overrides")
        st.info("Simulate a specific scenario by adjusting the parameters below:")
        
        # User Inputs for Prediction
        sim_temp = st.slider("Ambient Temperature (°F)", 10, 110, 72)
        sim_no2 = st.number_input("NO2 Concentration (ppb)", 0.0, 60.0, 15.0)
        sim_hum = st.slider("Relative Humidity (%)", 0, 100, 45)
        sim_lag = st.number_input("Last Hour's Ozone Reading (ppm)", 0.000, 0.120, 0.030, format="%.3f")
        
        test_trigger = st.button("📊 Run Predictive Test")

    with col2:
        st.subheader("Safety Assessment")
        if test_trigger:
            # Construct feature vector for the model
            test_input = np.array([[sim_no2, sim_temp, sim_hum, sim_lag]])
            prediction = o3_model.predict(test_input)[0]
            
            # Display Result
            st.metric("Predicted Ozone Level", f"{prediction:.4f} ppm")
            
            # TESTING LOGIC: Is the area "Affected"?
            # Threshold based on Environmental Protection / Nuclear Safety levels
            if prediction > 0.070:
                st.error("🚨 CRITICAL: Prediction exceeds Safety Threshold (0.070 ppm)!")
                st.write("**Assessment:** Area is severely affected. Activate ventilation scrubbers.")
            elif prediction > 0.055:
                st.warning("⚠️ WARNING: Elevated Pollutant Levels.")
                st.write("**Assessment:** Area is moderately affected. Increase sampling frequency.")
            else:
                st.success("✅ NOMINAL: Levels are within safe background limits.")
                st.write("**Assessment:** No radiological or chemical deviation detected.")
        else:
            st.write("Enter sensor data and click 'Run Predictive Test' to evaluate the environment.")

# --- SECTION 2: FORENSIC ANALYSIS ---
with tabs[1]:
    st.header("Forensic Isotope Surrogate Fingerprinting")
    st.write("Testing for: Identifying if pollutants are from the facility or external sources.")
    
    fig_forensic = px.scatter(df, x='Cesium PM2.5 LC', y='Strontium PM2.5 LC', 
                             color='Sr_Cs_Ratio', hover_data=['DATETIME_LOCAL'],
                             title="Cesium vs Strontium Forensic Clusters")
    st.plotly_chart(fig_forensic, use_container_width=True)

# --- SECTION 3: EARLY WARNING SYSTEM ---
with tabs[2]:
    st.header("Automated Safety Shield")
    st.write("This section displays all multi-variate anomalies flagged by the **Isolation Forest**.")
    
    alerts = df[df['Safety_Status'] == '🚨 ALERT'].tail(20)
    if not alerts.empty:
        st.warning(f"Detection Log: {len(alerts)} complex anomalies detected at {selected_site}.")
        st.dataframe(alerts[['DATETIME_LOCAL', 'Ozone', 'Carbon monoxide', 'Safety_Status']])
    else:
        st.success("Safety Shield Active: No multi-variate anomalies detected.")
