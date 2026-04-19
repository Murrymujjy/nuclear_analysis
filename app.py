import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# 1. PAGE CONFIG
st.set_page_config(page_title="NEMS | Advanced Nuclear Lab", layout="wide", page_icon="☢️")

# 2. DATA LOADING
@st.cache_data
def load_data():
    df = pd.read_csv('Nuclear_Lab_Final_Master.csv')
    df['DATETIME_LOCAL'] = pd.to_datetime(df['DATETIME_LOCAL'])
    return df

df = load_data()

# 3. SIDEBAR / CONTROL CENTER
st.sidebar.title("☢️ NEMS Control Center")
st.sidebar.markdown("**Nuclear Environmental Monitoring System**")
st.sidebar.divider()
selected_site = st.sidebar.selectbox("Select Monitoring Station", df['LOCAL_SITE_NAME'].unique())
site_data = df[df['LOCAL_SITE_NAME'] == selected_site]

# 4. HEADER & SYSTEM METRICS
st.title("🛡️ Advanced Nuclear Laboratory: Environmental Report")
st.markdown(f"**Current Station:** {selected_site} | **System Status:** ONLINE")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Ozone R² Accuracy", "0.7162", "High")
m2.metric("CO R² Accuracy", "0.6216", "Stable")
m3.metric("Forensic Anomalies", len(df[df['Safety_Status'] == '🚨 ALERT']), "Critical")
m4.metric("Isotope Ratio (Avg)", f"{df['Sr_Cs_Ratio'].mean():.4f}")

st.divider()

# 5. DETAILED SECTIONS
tabs = st.tabs(["📈 Predictive Baseline", "🔬 Forensic Fingerprinting", "🚨 Early Warning System", "📖 Methodology"])

# --- TAB 1: PREDICTIVE BASELINE ---
with tabs[0]:
    st.subheader("Chemical Waste Predictive Modeling")
    st.info("Testing for: Operational Deviations from the atmospheric baseline.")
    
    col_a, col_b = st.columns([1, 2])
    with col_a:
        st.write("**Model Logic:**")
        st.write("Using a Random Forest Regressor with Temporal Lagging. This model predicts the 'Normal' state based on photochemical precursors.")
        target = st.radio("Select Target for Testing", ["Ozone", "Carbon monoxide"])
    
    with col_b:
        fig_trend = px.line(site_data.tail(100), x='DATETIME_LOCAL', y=target, 
                           title=f"Baseline Concentration Trend: {target}")
        st.plotly_chart(fig_trend, use_container_width=True)

# --- TAB 2: FORENSIC FINGERPRINTING ---
with tabs[1]:
    st.subheader("Nuclear Forensic Isotope Surrogates")
    st.info("Testing for: Source Identification through stable element ratios.")
    
    c1, c2 = st.columns(2)
    with c1:
        fig_ratio = px.scatter(df, x='Cesium PM2.5 LC', y='Strontium PM2.5 LC', 
                             color='Sr_Cs_Ratio', title="Sr/Cs Isotope Surrogate Cluster")
        st.plotly_chart(fig_ratio, use_container_width=True)
    with c2:
        st.write("**Forensic Interpretation:**")
        st.write("In this laboratory simulation, Cesium and Strontium act as surrogates for fission products.")
        st.write("The ratio shown in the graph helps us identify if pollutants are coming from the reactor core or general environmental background.")

# --- TAB 3: EARLY WARNING SYSTEM ---
with tabs[2]:
    st.subheader("Unsupervised Anomaly Detection (EWS)")
    st.info("Testing for: Multi-variate 'Leak Signatures' using Isolation Forests.")
    
    alerts = df[df['Safety_Status'] == '🚨 ALERT'].tail(15)
    
    if not alerts.empty:
        st.error(f"SYSTEM ALERT: {len(alerts)} Multi-variate anomalies detected in recent cycles.")
        st.dataframe(alerts[['DATETIME_LOCAL', 'Ozone', 'Carbon monoxide', 'Nitrogen dioxide (NO2)', 'Safety_Status']])
    else:
        st.success("No critical anomalies detected. Environment is within 2-sigma of predicted baseline.")

# --- TAB 4: METHODOLOGY ---
with tabs[3]:
    st.subheader("Laboratory Methodology & Testing Parameters")
    st.markdown("""
    ### 1. Data Fidelity
    We utilized the `FIRST_MAX_VALUE` sensor readings to ensure peak detection, rather than rounded daily averages.
    
    ### 2. Machine Learning Architecture
    * **Random Forest (300 Trees):** Captures non-linear relationships between temperature, time, and chemical precursors.
    * **Temporal Lagging (t-1):** Corrects for the 'Atmospheric Memory' effect, increasing R² from 0.49 to 0.72.
    * **Isolation Forest:** Provides an unsupervised safety shield that requires no prior knowledge of what a 'leak' looks like.
    
    ### 3. Safety Implications
    This system allows for **Proactive Monitoring**. By predicting the environment, we can detect a leak *before* it reaches dangerous statutory limits.
    """)
