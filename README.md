# Agri Assistant API

A comprehensive agricultural prediction system with 4 ML-powered services:
1. **Crop Recommendation** - Recommends best crops based on soil and weather conditions
2. **Yield Prediction** - Predicts crop yield based on various factors
3. **Fertilizer Recommendation** - Suggests optimal fertilizers for your crops
4. **Weather Forecast** - Predicts rain/no rain based on weather conditions

## 🚀 Quick Start

### 1. Activate Virtual Environment
```bash
source venv/bin/activate
```

### 2. Install Dependencies (if not already installed)
```bash
pip install -r requirements.txt
```

### 3. Train Models (if not already trained)
```bash
python train_models.py
```

### 4. Start the API Server
```bash
python app.py
```

The API will be available at: `http://127.0.0.1:8000`

### 5. Open the Web Interface
Open `index.html` in your browser, or visit:
- API Docs: `http://127.0.0.1:8000/docs`
- Alternative Docs: `http://127.0.0.1:8000/redoc`

## 📁 Project Structure

```
crop/
├── app.py                      # FastAPI backend server
├── train_models.py             # ML model training script
├── index.html                  # Web interface (Swagger-style UI)
├── requirements.txt            # Python dependencies
├── models/                     # Trained ML models (generated)
│   ├── crop_recommend_model.pkl
│   ├── yield_model.pkl
│   ├── fertilizer_model.pkl
│   ├── weather_model.pkl
│   └── [encoders...]
├── Crop_recommendation.csv     # Dataset 1
├── Crop_Yield_Prediction.csv   # Dataset 2
├── Fertilizer Prediction.csv   # Dataset 3
└── weather_forecast_data.csv   # Dataset 4
```

## 🔌 API Endpoints

### 1. Crop Recommendation
**POST** `/crop_recommend`
```json
{
  "N": 90,
  "P": 50,
  "K": 40,
  "temperature": 25,
  "humidity": 65,
  "ph": 6.5,
  "rainfall": 120
}
```

### 2. Yield Prediction
**POST** `/yield_predict`
```json
{
  "Nitrogen": 90,
  "Phosphorus": 50,
  "Potassium": 40,
  "Temperature": 25,
  "Humidity": 65,
  "pH_Value": 6.5,
  "Rainfall": 120,
  "Crop": "Rice"
}
```

### 3. Fertilizer Recommendation
**POST** `/fertilizer_recommend`
```json
{
  "Temparature": 26,
  "Humidity": 52,
  "Moisture": 38,
  "Soil_Type": "Sandy",
  "Crop_Type": "Maize",
  "Nitrogen": 37,
  "Potassium": 0,
  "Phosphorous": 0
}
```

### 4. Weather Forecast
**POST** `/weather_forecast`
```json
{
  "Temperature": 23.72,
  "Humidity": 89.59,
  "Wind_Speed": 7.34,
  "Cloud_Cover": 50.50,
  "Pressure": 1032.38
}
```

## 🎯 Features

- ✅ 4 ML models trained on real agricultural data
- ✅ FastAPI backend with automatic API documentation
- ✅ Beautiful Swagger-style web interface
- ✅ Tabbed interface for easy navigation
- ✅ Real-time predictions
- ✅ CORS enabled for frontend integration

## 📊 Model Information

- **Crop Recommendation**: Random Forest Classifier (22 crop types)
- **Yield Prediction**: Random Forest Regressor
- **Fertilizer Recommendation**: Random Forest Classifier (7 fertilizer types)
- **Weather Forecast**: Random Forest Classifier (rain/no rain)

## 🛠️ Technologies Used

- Python 3.13
- FastAPI
- scikit-learn
- pandas
- numpy
- HTML/CSS/JavaScript

## 📝 Notes

- Models are trained using Random Forest algorithms
- All datasets have been cleaned and validated
- The web interface mimics Swagger UI design
- API supports CORS for cross-origin requests

## 🔧 Troubleshooting

If models are not loading:
1. Make sure `models/` directory exists
2. Run `python train_models.py` to regenerate models
3. Check that all CSV files are in the project root

If API server won't start:
1. Check if port 8000 is available
2. Make sure all dependencies are installed
3. Verify virtual environment is activated

