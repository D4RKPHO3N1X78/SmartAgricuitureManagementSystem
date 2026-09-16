import time
import numpy as np
import onnxruntime as ort
import joblib
import os
import json

def simulate_edge_device():
    onnx_model_path = "models/yield_model.onnx"
    scaler_path = "models/scaler.save"
    
    if not os.path.exists(onnx_model_path) or not os.path.exists(scaler_path):
        print("Error: Models not found. Train the model first.")
        return
        
    print("Loading ONNX model and Scaler...")
    session = ort.InferenceSession(onnx_model_path)
    scaler = joblib.load(scaler_path)
    
    input_name = session.get_inputs()[0].name
    
    print("\n--- Starting Edge Inference Simulation (ONNX Runtime) ---")
    try:
        while True:
            # Simulate randomized edge sensor readings (close to realistic bounds)
            temp = np.random.uniform(15, 35)
            hum = np.random.uniform(40, 90)
            ph = np.random.uniform(5.5, 7.5)
            rain = np.random.uniform(0, 150)
            n_val = np.random.uniform(20, 80)
            p_val = np.random.uniform(10, 60)
            k_val = np.random.uniform(10, 50)
            
            raw_input = np.array([[temp, hum, ph, rain, n_val, p_val, k_val]])
            
            # Preprocess using the loaded scaler
            processed_input = scaler.transform(raw_input).astype(np.float32)
            
            # Start timer for inference speed
            start_t = time.time()
            
            # Run inference
            result = session.run(None, {input_name: processed_input})
            yield_pred = result[0][0][0]
            
            inf_time_ms = (time.time() - start_t) * 1000
            
            # Output payload
            payload = {
                "sensors": {
                    "Temp(C)": round(temp, 1),
                    "Humidity(%)": round(hum, 1),
                    "Soil_pH": round(ph, 2),
                    "Rainfall(mm)": round(rain, 1),
                    "NPK": [round(n_val, 1), round(p_val, 1), round(k_val, 1)]
                },
                "prediction": {
                    "Yield_Tonnes_per_Ha": round(float(yield_pred), 2),
                    "Inference_Time_ms": round(inf_time_ms, 2)
                }
            }
            
            print(json.dumps(payload, indent=2))
            
            # Simulate interval between readings (e.g., 3 seconds)
            time.sleep(3)
            
    except KeyboardInterrupt:
        print("\nSimulation stopped.")

if __name__ == "__main__":
    simulate_edge_device()
