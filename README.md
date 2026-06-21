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
* **Machine Learning & Inference:** ONNX Runtime (`onnxruntime` for lightweight production inference), XGBoost Regressor (training), Scikit-Learn, Pandas, NumPy, ONNXMLTools (conversion).

---

## 📁 Repository Structure

```text
├── backend/
│   ├── impute_data.py          # Data cleaning & mode-imputation pipeline
│   ├── main.py                 # FastAPI backend server & inference endpoints
│   ├── optimizer.py            # Nearest-Neighbor TSP route optimization logic
│   ├── parking_ml_model.onnx   # Trained model exported to ONNX format
│   ├── parking_ml_model_meta.json # Runtime metadata (features, classes, clusters)
│   ├── overall_stats.json      # Pre-cached statistical summaries of dataset
│   ├── hotspot_clusters.json   # Pre-calculated DBSCAN cluster centers
│   └── locations_db.json       # Predefined directory of junctions & coordinates
├── frontend/
│   ├── index.html              # Main web dashboard interface layout
│   ├── styles.css              # Dashboard styling & Leaflet animations
│   └── app.js                  # Frontend state, map rendering, and slider logic
├── xgb/
│   ├── train_xgb.py            # XGBoost training & validation splits pipeline
│   ├── xgb_model.pkl           # Raw trained regressor model output (local backup)
│   ├── xgb_meta.pkl            # Trained model label encoders & R² logs (local backup)
│   └── experiment_results.json # Metric results of split experiments
├── requirements-train.txt      # Model training & export dependencies
├── requirements.txt            # Production dependencies (ONNX Runtime, FastAPI, etc.)
└── README.md                   # Project documentation
```

---

## ⚙️ Installation & Setup

### 1. Prerequisites
Ensure Python 3.9+ is installed (Python 3.11/3.12 recommended).

### 2. Install Dependencies

Depending on your environment, you can install lightweight dependencies for production runtime or full dependencies for model training/export.

#### Production Runtime (Lighter Bundle for Serverless/FastAPI)
Installs `onnxruntime`, `numpy`, `fastapi`, and `uvicorn` (no heavy `xgboost` or `scikit-learn` libraries needed, which fits safely within Vercel's 500MB serverless limit):
```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`

# Install runtime packages
pip install -r requirements.txt
```

#### Training & Development (Required to train or convert the model)
Installs full ML libraries (`xgboost`, `scikit-learn`, `pandas`, `onnxmltools`, `onnx`):
```bash
pip install -r requirements-train.txt
```

### 3. Data Processing & Imputation (Optional)
If running from raw data, clean and impute missing variables in the dataset:
```bash
python backend/impute_data.py
```

### 4. Train and Export the Model to ONNX
To retrain the model on the timezone-corrected Bengaluru local time dataset, convert it to ONNX, and generate the JSON metadata:
```bash
# Ensure training dependencies are installed first
pip install -r requirements-train.txt

# Run the training pipeline
python xgb/train_xgb.py
```
This script saves the model as `backend/parking_ml_model.onnx` and writes features, vehicle classes, and cluster centers to `backend/parking_ml_model_meta.json`.

### 5. Run the Dashboard Local Server
Start the Uvicorn server to host the web dashboard locally:
```bash
# Start server
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```
Once running, open your web browser and navigate to:
👉 **[http://127.0.0.1:8000](http://127.0.0.1:8000)**

---

## 📊 Model Performance Validation

The XGBoost model is validated under two splits to ensure robust generalizations:
* **Random Train/Test Split:** $R^2$ Score of **0.8818** on test subset.
* **Chronological Split (Temporal Validation):** $R^2$ Score of **0.8562** on test subset.
