from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import os
import numpy as np

app = FastAPI(title="Agri Assistant API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models and encoders
models = {}
encoders = {}

def load_models():
    """Load all trained models and encoders"""
    try:
        # Crop Recommendation
        if os.path.exists('models/crop_recommend_model.pkl'):
            with open('models/crop_recommend_model.pkl', 'rb') as f:
                models['crop_recommend'] = pickle.load(f)
        
        # Yield Prediction
        if os.path.exists('models/yield_model.pkl'):
            with open('models/yield_model.pkl', 'rb') as f:
                models['yield'] = pickle.load(f)
            with open('models/crop_encoder.pkl', 'rb') as f:
                encoders['crop'] = pickle.load(f)
        
        # Fertilizer Recommendation
        if os.path.exists('models/fertilizer_model.pkl'):
            with open('models/fertilizer_model.pkl', 'rb') as f:
                models['fertilizer'] = pickle.load(f)
            with open('models/soil_encoder.pkl', 'rb') as f:
                encoders['soil'] = pickle.load(f)
            with open('models/croptype_encoder.pkl', 'rb') as f:
                encoders['croptype'] = pickle.load(f)
            with open('models/fertilizer_encoder.pkl', 'rb') as f:
                encoders['fertilizer'] = pickle.load(f)
        
        # Weather Forecast
        if os.path.exists('models/weather_model.pkl'):
            with open('models/weather_model.pkl', 'rb') as f:
                models['weather'] = pickle.load(f)
            with open('models/rain_encoder.pkl', 'rb') as f:
                encoders['rain'] = pickle.load(f)
        
        print("✅ All models loaded successfully!")
    except Exception as e:
        print(f"⚠️  Error loading models: {e}")

# Load models on startup
load_models()

# Request Models
class CropRecommendInput(BaseModel):
    N: float
    P: float
    K: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float

class YieldPredictionInput(BaseModel):
    Nitrogen: float
    Phosphorus: float
    Potassium: float
    Temperature: float
    Humidity: float
    pH_Value: float
    Rainfall: float
    Crop: str

class FertilizerInput(BaseModel):
    Temparature: float
    Humidity: float
    Moisture: float
    Soil_Type: str
    Crop_Type: str
    Nitrogen: float
    Potassium: float
    Phosphorous: float

class WeatherInput(BaseModel):
    Temperature: float
    Humidity: float
    Wind_Speed: float
    Cloud_Cover: float
    Pressure: float

@app.get("/")
def read_root():
    return {
        "message": "Agri Assistant API",
        "endpoints": {
            "crop_recommend": "/crop_recommend",
            "yield_predict": "/yield_predict",
            "fertilizer_recommend": "/fertilizer_recommend",
            "weather_forecast": "/weather_forecast"
        }
    }

@app.post("/crop_recommend")
def crop_recommend(data: CropRecommendInput):
    """Recommend crop based on soil and weather conditions"""
    if 'crop_recommend' not in models:
        raise HTTPException(status_code=500, detail="Crop recommendation model not loaded")
    
    try:
        input_data = np.array([[data.N, data.P, data.K, data.temperature, 
                               data.humidity, data.ph, data.rainfall]])
        prediction = models['crop_recommend'].predict(input_data)[0]
        return {"recommended_crop": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/yield_predict")
def yield_predict(data: YieldPredictionInput):
    """Predict crop yield based on conditions"""
    if 'yield' not in models:
        raise HTTPException(status_code=500, detail="Yield prediction model not loaded")
    
    try:
        # Encode crop name
        if data.Crop not in encoders['crop'].classes_:
            raise HTTPException(status_code=400, detail=f"Unknown crop: {data.Crop}")
        
        crop_encoded = encoders['crop'].transform([data.Crop])[0]
        input_data = np.array([[data.Nitrogen, data.Phosphorus, data.Potassium, 
                               data.Temperature, data.Humidity, data.pH_Value, 
                               data.Rainfall, crop_encoded]])
        prediction = models['yield'].predict(input_data)[0]
        return {"predicted_yield": round(float(prediction), 2), "unit": "kg/hectare"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/fertilizer_recommend")
def fertilizer_recommend(data: FertilizerInput):
    """Recommend fertilizer based on soil and crop conditions"""
    if 'fertilizer' not in models:
        raise HTTPException(status_code=500, detail="Fertilizer recommendation model not loaded")
    
    try:
        # Encode categorical features
        if data.Soil_Type not in encoders['soil'].classes_:
            raise HTTPException(status_code=400, detail=f"Unknown soil type: {data.Soil_Type}")
        if data.Crop_Type not in encoders['croptype'].classes_:
            raise HTTPException(status_code=400, detail=f"Unknown crop type: {data.Crop_Type}")
        
        soil_encoded = encoders['soil'].transform([data.Soil_Type])[0]
        croptype_encoded = encoders['croptype'].transform([data.Crop_Type])[0]
        input_data = np.array([[data.Temparature, data.Humidity, data.Moisture, 
                               soil_encoded, croptype_encoded, data.Nitrogen, 
                               data.Potassium, data.Phosphorous]])
        prediction_encoded = models['fertilizer'].predict(input_data)[0]
        prediction = encoders['fertilizer'].inverse_transform([prediction_encoded])[0]
        return {"recommended_fertilizer": prediction}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/weather_forecast")
def weather_forecast(data: WeatherInput):
    """Forecast weather (rain/no rain) based on conditions"""
    if 'weather' not in models:
        raise HTTPException(status_code=500, detail="Weather forecast model not loaded")
    
    try:
        input_data = np.array([[data.Temperature, data.Humidity, data.Wind_Speed, 
                               data.Cloud_Cover, data.Pressure]])
        prediction_encoded = models['weather'].predict(input_data)[0]
        prediction = encoders['rain'].inverse_transform([prediction_encoded])[0]
        return {"forecast": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    print("\n🚀 Starting Agri Assistant API...")
    print("📡 Server running at http://127.0.0.1:8000")
    print("📚 API docs at http://127.0.0.1:8000/docs\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)

