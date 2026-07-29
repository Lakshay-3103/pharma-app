from fastapi import FastAPI, HTTPException, Query
from contextlib import asynccontextmanager
import pandas as pd
import joblib

# Global dictionaries to hold our data and ML models
app_data = {}
ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting up: Loading models and dataset...")
    try:
        # Using absolute Colab paths to ensure they are always found
        ml_models['rf_model'] = joblib.load('model/medicine_price_model.pkl')
        ml_models['manuf_encoder'] = joblib.load('model/manuf_encoder.pkl')
        ml_models['comp_encoder'] = joblib.load('model/comp_encoder.pkl')
        app_data['df'] = pd.read_csv('data/cleaned_medicines.csv')
        print("Startup complete! Ready to serve requests.")
    except Exception as e:
        print(f"CRITICAL ERROR during startup: {e}")
    yield 
    app_data.clear()
    ml_models.clear()

app = FastAPI(lifespan=lifespan, title="Medicine Price Predictor API")

@app.get("/predict-price")
async def predict_price(medicine_name: str = Query(..., description="The brand name of the medicine")):
    df = app_data.get('df')
    
    # Safety check if dataframe failed to load
    if df is None:
        raise HTTPException(status_code=500, detail="Server Error: Dataset not loaded into memory.")
        
    medicine_data = df[df['brand_name'].str.lower() == medicine_name.lower()]
    
    if medicine_data.empty:
        raise HTTPException(status_code=404, detail=f"Medicine '{medicine_name}' not found.")
        
    med_info = medicine_data.iloc[0]
    manufacturer = med_info['manufacturer']
    composition = med_info['cleaned_composition']
    
    try:
        manuf_encoded = ml_models['manuf_encoder'].transform([manufacturer])[0]
        comp_encoded = ml_models['comp_encoder'].transform([composition])[0]
        prediction = ml_models['rf_model'].predict([[manuf_encoded, comp_encoded]])
        return {"predicted_price": round(prediction[0], 2)}
    except ValueError:
         raise HTTPException(status_code=500, detail="Error: Encountered unknown data.")

@app.get("/alternatives")
async def get_alternatives(medicine_name: str = Query(..., description="The brand name of the medicine")):
    df = app_data.get('df')
    
    if df is None:
         raise HTTPException(status_code=500, detail="Server Error: Dataset not loaded into memory.")
         
    medicine_data = df[df['brand_name'].str.lower() == medicine_name.lower()]
    
    if medicine_data.empty:
        raise HTTPException(status_code=404, detail=f"Medicine '{medicine_name}' not found.")
        
    med_info = medicine_data.iloc[0]
    target_comp = med_info['cleaned_composition']
    current_price = med_info['price_inr']
    
    matches = df[df['cleaned_composition'] == target_comp]
    cheaper_options = matches[matches['price_inr'] < current_price].sort_values(by='price_inr', ascending=True)
    
    display_columns = ['brand_name', 'manufacturer', 'pack_size', 'pack_unit', 'price_inr']
    results = cheaper_options[display_columns].head(5)
    
    return {"alternatives": results.to_dict(orient='records')}
