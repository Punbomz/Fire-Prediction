# Ignis AI: Wildfire Risk Prediction & Visualization Dashboard

**Ignis AI** is an end-to-end geospatial intelligence platform designed to predict and visualize wildfire risks. The project integrates satellite imagery analysis, machine learning inference, and interactive web visualization to provide a real-time simulation tool for wildfire management.

## 🚀 Project Pipeline

The project is divided into three main phases:

### 1. Data Engineering & Preprocessing (`/notebooks`, `/data`)
This phase focuses on transforming raw spatial and environmental data into a machine-learning-ready format.
*   **Feature Extraction:** Extracting spectral indices (NDVI, NDWI, NBR) from Sentinel-2/Landsat imagery and integrating topographic data (Elevation, Slope).
*   **Baseline Analysis:** Generating a `baseline_table.csv` to establish monthly historical norms for each province, allowing the system to calculate **Spatial Anomalies**.
*   **Spatial Sampling:** Sampling geographic points within administrative boundaries (Districts) to represent continuous environmental surfaces.

### 2. Machine Learning & Model Training
*   **Model:** A **CatBoost Classifier** trained on historical wildfire occurrences and environmental features.
*   **Spatial Clustering:** Utilizing **K-Means Clustering** to categorize geographic regions based on topographic and vegetation similarities.
*   **Optimization:** Feature engineering of 32 distinct variables, including vegetation stress, terrain roughness, and hot-dry stress indices.

### 3. Web Application & Inference Deployment
The system utilizes a **Decoupled Architecture** for high performance and scalability:
*   **Frontend (Next.js):** An interactive React dashboard utilizing **React-Leaflet** for geospatial rendering. Features include a real-time simulation panel for temperature and seasonality adjustments.
*   **Orchestration Layer (Next.js API):** Acts as a bridge between the map and the AI, managing point-data retrieval and coordinating with the inference server.
*   **Inference Engine (Flask + Render):** A dedicated Python server that hosts the CatBoost model. It performs real-time feature engineering (Anomaly calculation, Clustering) and returns risk probabilities.

---

## 🗺️ Data
https://drive.google.com/drive/folders/1OAKEuvwugnPz5XX0t4xYq7foX6mvUqbV?usp=sharing

## 📂 Project Structure (Wild-Fire-Risk-Prediction Repository)
https://github.com/Punbomz/Wild-Fire-Risk-Prediction

```bash
├── data/               # Geospatial datasets (Baseline CSV, District points JSON)
├── models/             # Trained AI models (CatBoost .cbm, K-Means .pkl)
├── notebooks/          # Data processing and model training scripts
├── pages/api/          # Next.js API Routes (Backend Orchestration)
├── components/         # React UI Components (Map, Sidebar, Filters)
├── app.py              # Flask Inference Server (AI Backend)
└── requirements.txt    # Python dependencies
```

---

## 🛠️ Key Features

-   **Interactive Risk Map:** Visualize fire risks across provinces and districts with dynamic color-coding (Green to Red).
-   **"What-If" Simulation:** Manually adjust temperature and month parameters to see how environmental shifts impact wildfire probability.
-   **Anomaly Detection:** The AI identifies risk based on deviations from historical norms, not just absolute values.
-   **Point-Level Analytics:** Detailed tooltips for each spatial point, showing confidence scores and specific environmental metrics.

## ⚙️ Setup & Installation

### Backend (AI Server)
```bash
pip install -r requirements.txt
python app.py
```

### Frontend (Dashboard)
```bash
npm install
npm run dev
```

---

## 🧠 Developed By
*   **Project Name:** Wildfire Risk Visualization (Ignis AI)
*   **Focus Area:** GIS, Machine Learning, Web Development
