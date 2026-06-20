# 🏙️ APCI - AI Parking Congestion Intelligence

APCI (**AI Parking Congestion Intelligence**) is an advanced machine learning forecasting system and dynamic analytics dashboard designed to predict, visualize, and optimize response plans for illegal parking violations and traffic congestion in Bengaluru.

The platform utilizes a timezone-aligned **XGBoost Regressor model** trained on spatial-temporal data (~291k records) to forecast real-time traffic impact scores, maps hotspots via DBSCAN spatial clustering, and computes optimized patrol routes for traffic enforcement teams.

---

## 🚀 Key Features

* **Real-time Traffic Impact Prediction (ML):** XGBoost regressor model predicting local traffic congestion index (0–100%) on the fly based on vehicle weight, location coordinates, hour, weekday, and proximity to major Points of Interest (POIs) such as Metro stations and Commercial hubs.
* **Geospatial Hotspot Map (DBSCAN):** Dynamic visualization map using **Leaflet.js** identifying clustering centers of illegal parking violations (hotspots), utilizing baseline density values scaling seamlessly relative to timeline adjustments.
* **Continuous Timeline Slider:** Smooth linear-interpolation timeline slider representing dynamic day-scenario metrics (expected hourly violations mapped to local IST Bengaluru time).
* **Patrol Route Optimizer:** Dynamic enforcement routing module utilizing the **Nearest Neighbor Traveling Salesperson (TSP)** heuristic to calculate the shortest path through top hotspots.
* **Jurisdiction Leaderboard:** Analytical leaderboard breaking down violation statistics, average traffic impact scores, and recurrence metrics by police station jurisdictions.

---

## 🛠️ Technology Stack

* **Frontend:** Vanilla HTML5, CSS3 (Modern flat traffic-monitoring UI theme with glassmorphism), Leaflet.js (Map), Chart.js (Timeline Trends & POI scatter plot).
* **Backend:** FastAPI (Python), Uvicorn.
* **Machine Learning:** XGBoost Regressor, Scikit-Learn (Label Encoding, Split Validation), Pandas, NumPy.

---

## 📁 Repository Structure

```text
├── backend/
│   ├── impute_data.py          # Data cleaning & mode-imputation pipeline
│   ├── main.py                 # FastAPI backend server & inference endpoints
│   ├── optimizer.py            # Nearest-Neighbor TSP route optimization logic
│   ├── parking_ml_model.pkl    # Final packaged model loaded by FastAPI
│   ├── overall_stats.json      # Pre-cached statistical summaries of dataset
│   ├── hotspot_clusters.json   # Pre-calculated DBSCAN cluster centers
│   └── locations_db.json       # Predefined directory of junctions & coordinates
├── frontend/
│   ├── index.html              # Main web dashboard interface layout
│   ├── styles.css              # Dashboard styling & Leaflet animations
│   └── app.js                  # Frontend state, map rendering, and slider logic
├── xgb/
│   ├── train_xgb.py            # XGBoost training & validation splits pipeline
│   ├── xgb_model.pkl           # Raw trained regressor model output
│   ├── xgb_meta.pkl            # Trained model label encoders & R² logs
│   └── experiment_results.json # Metric results of split experiments
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

---

## ⚙️ Installation & Setup

### 1. Prerequisites
Ensure Python 3.9+ is installed.

### 2. Install Dependencies
Initialize a virtual environment and install the required libraries:
```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`

# Install required packages
pip install -r requirements.txt
```

### 3. Data Processing & Imputation (Optional)
If running from raw data, clean and impute missing variables in the dataset:
```bash
python backend/impute_data.py
```

### 4. Train the XGBoost Model
Retrain the model on the timezone-corrected Bengaluru local time dataset:
```bash
python xgb/train_xgb.py
```

### 5. Run the Dashboard Local Server
Start the Uvicorn server to host the web dashboard locally:
```bash
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```
Once running, open your web browser and navigate to:
👉 **[http://127.0.0.1:8000](http://127.0.0.1:8000)**

---

## 📊 Model Performance Validation

The XGBoost model is validated under two splits to ensure robust generalizations:
* **Random Train/Test Split:** $R^2$ Score of **0.8818** on test subset.
* **Chronological Split (Temporal Validation):** $R^2$ Score of **0.8562** on test subset.
