# nuclear_analysis

# 🛡️ NEMS: Nuclear Environmental Monitoring System
**Advanced Nuclear Laboratory | Course Project**

## 📌 Project Overview
The **Nuclear Environmental Monitoring System (NEMS)** is a Python-based decision support tool designed to monitor chemical waste and atmospheric conditions around a nuclear facility. By leveraging Machine Learning and Nuclear Forensic proxies, NEMS differentiates between natural environmental fluctuations and potential industrial anomalies.

## 🚀 Key Features
* **Predictive Baseline Modeling:** Uses a Random Forest Regressor ($R^2 = 0.7162$) to predict expected Ozone levels based on photochemical precursors and temporal lagging.
* **Interactive Simulation Sandbox:** Allows technicians to input manual sensor data to test if current conditions exceed safety thresholds (0.070 ppm).
* **Nuclear Forensic Fingerprinting:** Analyzes stable isotope surrogate ratios ($Sr/Cs$ and $Pb/Br$) to identify the source of pollutants.
* **Early Warning System (EWS):** Employs an unsupervised Isolation Forest to detect "statistically weird" multi-variate anomalies that traditional threshold alarms miss.

## 🛠️ Tech Stack
* **Language:** Python 3.10+
* **Framework:** Streamlit (Web Interface)
* **Data Science:** Pandas, NumPy, Scikit-Learn
* **Visualization:** Plotly, Seaborn

## 🧪 The Science Behind NEMS
### 1. Temporal Lagging
Atmospheric chemistry has "memory." NEMS incorporates **Lag-1 features** (the concentration from the previous hour), which improved our model accuracy from 49% to 72%. This accounts for the persistence of chemical plumes.

### 2. Isotope Surrogacy
In this laboratory simulation, stable **Strontium (Sr)** and **Cesium (Cs)** are used as proxies for the fission products $Sr-90$ and $Cs-137$. Monitoring the ratio between these elements allows for forensic identification of the "Source Term."

## 📦 Installation & Setup
1. **Clone the repository or download the files:**
   `app.py`, `Nuclear_Lab_Final_Master.csv`, and `requirements.txt`.

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt


Evaluation Metrics
Ozone Prediction R²: 0.7162

Carbon Monoxide R²: 0.6216

Mean Absolute Error (MAE): 0.0053 ppm

Total Anomalies Detected: 65
