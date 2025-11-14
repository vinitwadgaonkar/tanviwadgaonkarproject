import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import pickle
import os

print("=" * 60)
print("Training Agri Assistant ML Models")
print("=" * 60)

os.makedirs('models', exist_ok=True)

# 1. Crop Recommendation Model
print("\n1. Training Crop Recommendation Model...")
df1 = pd.read_csv('Crop_recommendation.csv')
X1 = df1[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y1 = df1['label']
model1 = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model1.fit(X1, y1)
pickle.dump(model1, open('models/crop_recommend_model.pkl', 'wb'))
print(f"   ✓ Model trained on {len(df1)} samples")
print(f"   ✓ Unique crops: {y1.nunique()}")
print("   ✓ Crop Recommendation model saved")

# 2. Yield Prediction Model
print("\n2. Training Yield Prediction Model...")
df2 = pd.read_csv('Crop_Yield_Prediction.csv')
# Encode crop names
le_crop = LabelEncoder()
df2['Crop_encoded'] = le_crop.fit_transform(df2['Crop'])
X2 = df2[['Nitrogen', 'Phosphorus', 'Potassium', 'Temperature', 'Humidity', 'pH_Value', 'Rainfall', 'Crop_encoded']]
y2 = df2['Yield']
model2 = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model2.fit(X2, y2)
pickle.dump(model2, open('models/yield_model.pkl', 'wb'))
pickle.dump(le_crop, open('models/crop_encoder.pkl', 'wb'))
print(f"   ✓ Model trained on {len(df2)} samples")
print(f"   ✓ Unique crops: {df2['Crop'].nunique()}")
print(f"   ✓ Yield range: {y2.min():.2f} - {y2.max():.2f}")
print("   ✓ Yield Prediction model saved")

# 3. Fertilizer Recommendation Model
print("\n3. Training Fertilizer Recommendation Model...")
df3 = pd.read_csv('Fertilizer Prediction.csv')
# Clean column names (remove spaces)
df3.columns = df3.columns.str.strip()
# Encode categorical features
le_soil = LabelEncoder()
le_crop_type = LabelEncoder()
le_fertilizer = LabelEncoder()
df3['Soil_encoded'] = le_soil.fit_transform(df3['Soil Type'])
df3['CropType_encoded'] = le_crop_type.fit_transform(df3['Crop Type'])
df3['Fertilizer_encoded'] = le_fertilizer.fit_transform(df3['Fertilizer Name'])
X3 = df3[['Temparature', 'Humidity', 'Moisture', 'Soil_encoded', 'CropType_encoded', 'Nitrogen', 'Potassium', 'Phosphorous']]
y3 = df3['Fertilizer_encoded']
model3 = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model3.fit(X3, y3)
pickle.dump(model3, open('models/fertilizer_model.pkl', 'wb'))
pickle.dump(le_soil, open('models/soil_encoder.pkl', 'wb'))
pickle.dump(le_crop_type, open('models/croptype_encoder.pkl', 'wb'))
pickle.dump(le_fertilizer, open('models/fertilizer_encoder.pkl', 'wb'))
print(f"   ✓ Model trained on {len(df3)} samples")
print(f"   ✓ Unique fertilizers: {df3['Fertilizer Name'].nunique()}")
print(f"   ✓ Unique soil types: {df3['Soil Type'].nunique()}")
print("   ✓ Fertilizer Recommendation model saved")

# 4. Weather Forecast Model
print("\n4. Training Weather Forecast Model...")
df4 = pd.read_csv('weather_forecast_data.csv')
le_rain = LabelEncoder()
df4['Rain_encoded'] = le_rain.fit_transform(df4['Rain'])
X4 = df4[['Temperature', 'Humidity', 'Wind_Speed', 'Cloud_Cover', 'Pressure']]
y4 = df4['Rain_encoded']
model4 = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model4.fit(X4, y4)
pickle.dump(model4, open('models/weather_model.pkl', 'wb'))
pickle.dump(le_rain, open('models/rain_encoder.pkl', 'wb'))
print(f"   ✓ Model trained on {len(df4)} samples")
print(f"   ✓ Rain classes: {df4['Rain'].value_counts().to_dict()}")
print("   ✓ Weather Forecast model saved")

print("\n" + "=" * 60)
print("✅ All models trained and saved successfully!")
print("=" * 60)
print("\nModels saved in 'models/' directory:")
print("  - crop_recommend_model.pkl")
print("  - yield_model.pkl + crop_encoder.pkl")
print("  - fertilizer_model.pkl + encoders")
print("  - weather_model.pkl + rain_encoder.pkl")

