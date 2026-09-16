from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import onnxruntime as ort
import joblib
import numpy as np
import os
import time
from datetime import datetime

app = FastAPI(title="Smart Agriculture Yield Predictor")

# Load Models on start
onnx_model_path = os.path.join(os.path.dirname(__file__), "../models/yield_model.onnx")
scaler_path = os.path.join(os.path.dirname(__file__), "../models/scaler.save")

try:
    session = ort.InferenceSession(onnx_model_path)
    scaler = joblib.load(scaler_path)
    input_name = session.get_inputs()[0].name
except Exception as e:
    print(f"Failed to load models: {e}")
    session, scaler, input_name = None, None, None

# In-memory history of predictions for the dashboard analytics
prediction_history = []

class SensorPayload(BaseModel):
    temperature: float
    humidity: float
    soil_ph: float
    rainfall: float
    nitrogen: float
    phosphorus: float
    potassium: float

@app.post("/predict")
async def predict_yield(payload: SensorPayload):
    if session is None or scaler is None:
        raise HTTPException(status_code=500, detail="Models not loaded properly check backend logs.")
        
    start_t = time.time()
    
    raw_input = np.array([[
        payload.temperature, 
        payload.humidity, 
        payload.soil_ph, 
        payload.rainfall, 
        payload.nitrogen, 
        payload.phosphorus, 
        payload.potassium
    ]])
    
    try:
        processed_input = scaler.transform(raw_input).astype(np.float32)
        result = session.run(None, {input_name: processed_input})
        yield_pred = float(result[0][0][0])
        inf_time_ms = round((time.time() - start_t) * 1000, 2)
        
        # Log to in-memory history
        prediction_history.append({
            "timestamp": datetime.now().isoformat(),
            "inputs": payload.dict(),
            "yield_pred": round(yield_pred, 2)
        })
        
        # Prevent memory leaks by capping at 1000 items
        if len(prediction_history) > 1000:
            prediction_history.pop(0)
            
        return {
            "prediction_tonnes_ha": round(yield_pred, 2),
            "inference_time_ms": inf_time_ms
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analytics")
async def get_analytics():
    # Return aggregated statistics and raw history for frontend charting
    total_inferences = len(prediction_history)
    avg_yield = 0.0
    
    if total_inferences > 0:
        avg_yield = sum([item['yield_pred'] for item in prediction_history]) / total_inferences
        
    return {
        "summary": {
            "totalInferences": total_inferences,
            "avgYield": round(avg_yield, 2)
        },
        "history": prediction_history[-50:] # Provide the last 50 points to save network payload
    }

# Mount Frontend static files
frontend_dir = os.path.join(os.path.dirname(__file__), "../frontend")
app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

@app.get("/")
def read_root():
    return FileResponse(os.path.join(frontend_dir, "index.html"))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
