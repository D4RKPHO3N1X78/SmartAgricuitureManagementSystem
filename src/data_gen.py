import pandas as pd
import numpy as np
import os

def generate_data(num_samples=5000, output_path="data/dataset.csv"):
    np.random.seed(42)
    
    # Generate synthetic input features
    temperature = np.random.normal(loc=25, scale=5, size=num_samples) # degrees Celsius
    humidity = np.random.normal(loc=60, scale=15, size=num_samples) # %
    soil_ph = np.random.normal(loc=6.5, scale=0.5, size=num_samples)
    rainfall = np.random.normal(loc=100, scale=30, size=num_samples) # mm
    nitrogen = np.random.normal(loc=50, scale=15, size=num_samples)
    phosphorus = np.random.normal(loc=40, scale=10, size=num_samples)
    potassium = np.random.normal(loc=30, scale=8, size=num_samples)
    
    # Clip realistic values
    temperature = np.clip(temperature, 10, 45)
    humidity = np.clip(humidity, 20, 100)
    soil_ph = np.clip(soil_ph, 4.5, 8.5)
    rainfall = np.clip(rainfall, 0, 300)
    nitrogen = np.clip(nitrogen, 0, 100)
    phosphorus = np.clip(phosphorus, 0, 100)
    potassium = np.clip(potassium, 0, 100)
    
    # Calculate synthetic yield (mostly influenced by interacting factors)
    # This formula is arbitrary but complex enough for a DNN to learn
    yield_base = 5.0 + \
                 0.1 * temperature - 0.005 * (temperature - 25)**2 + \
                 0.05 * humidity + \
                 2.0 * (soil_ph - 4.5) - 0.5 * (soil_ph - 6.5)**2 + \
                 0.02 * rainfall + \
                 0.01 * nitrogen + 0.01 * phosphorus + 0.005 * potassium
                 
    # Add some noise
    yield_noise = np.random.normal(0, 0.5, size=num_samples)
    crop_yield = np.clip(yield_base + yield_noise, 1.0, 20.0) # Tonnes per hectare
    
    df = pd.DataFrame({
        'Temperature': temperature,
        'Humidity': humidity,
        'Soil_pH': soil_ph,
        'Rainfall': rainfall,
        'Nitrogen': nitrogen,
        'Phosphorus': phosphorus,
        'Potassium': potassium,
        'Yield': crop_yield
    })
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Dataset generated at {output_path} with {num_samples} samples.")

if __name__ == "__main__":
    generate_data()
