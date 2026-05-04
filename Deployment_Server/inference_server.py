from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import catboost as cb
import os
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. รายชื่อฟีเจอร์ตามลำดับเป๊ะๆ (อ้างอิงจาก model_d_features.csv)
FEATURE_NAMES = [
    'ndvi', 'ndwi', 'nbr', 'blue', 'green', 'red', 'nir', 'swir1', 'swir2',
    'temp', 'soil_moisture', 'wind_u', 'wind_v', 'elev', 'slope', 'aspect',
    'landcover', 'month', 'NAME_1', 'NAME_2', 'veg_stress', 'fire_weather_idx',
    'wind_speed', 'drought_proxy', 'terrain_roughness', 'hot_dry_stress',
    'ndvi_anomaly', 'moisture_anomaly', 'temp_anomaly', 'month_sin', 'month_cos',
    'cluster_id'
]

MODEL_PATH = "wildfire_improved_model_d.cbm"
model = None

if os.path.exists(MODEL_PATH):
    model = cb.CatBoostClassifier()
    model.load_model(MODEL_PATH)
    print(f"✅ Model loaded successfully. Ready for {len(FEATURE_NAMES)} features.")
else:
    print(f"⚠️ Warning: Model file {MODEL_PATH} not found!")

class PredictionRequest(BaseModel):
    features: list 

@app.post("/predict")
async def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        if len(request.features) != len(FEATURE_NAMES):
             raise ValueError(f"Expected {len(FEATURE_NAMES)} features, but got {len(request.features)}")

        # สร้าง DataFrame พร้อมระบุชื่อคอลัมน์ให้ตรงกับที่ CatBoost ต้องการ
        df = pd.DataFrame([request.features], columns=FEATURE_NAMES)
        
        # ทำนายผล
        prob = model.predict_proba(df)[0][1]
        
        return {
            "probability": float(prob),
            "status": "success"
        }
    except Exception as e:
        print(f"❌ Prediction error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
