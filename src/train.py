import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
import joblib
import os
import mlflow
import mlflow.sklearn
import mlflow.onnx
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

def build_and_train(dataset_path="data/dataset.csv", models_dir="models"):
    print("Loading dataset...")
    df = pd.read_csv(dataset_path)
    
    X = df.drop('Yield', axis=1)
    y = df['Yield']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)
    
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(scaler, os.path.join(models_dir, "scaler.save"))
    
    # MLflow automatically tracks parameters, metrics, and models for sklearn
    mlflow.sklearn.autolog()
    
    # Optional: explicitly set experiment name
    mlflow.set_experiment("Crop_Yield_Prediction")
    
    with mlflow.start_run(run_name="MLP_Regressor_Run"):
        print("Training model...")
        model = MLPRegressor(hidden_layer_sizes=(32, 16), activation='relu', max_iter=500, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # Predict and evaluate
        y_pred = model.predict(X_test_scaled)
        mae = np.mean(np.abs(y_test - y_pred))
        print(f"Test MAE: {mae:.2f}")
        
        # Log manual metric just to be explicit
        mlflow.log_metric("test_mae", mae)
        
        # Save local artifacts like we did previously
        sklearn_model_path = os.path.join(models_dir, "yield_model.joblib")
        joblib.dump(model, sklearn_model_path)
        print(f"Saved joblib model to {sklearn_model_path}")
        
        print("Converting to ONNX...")
        initial_type = [('float_input', FloatTensorType([None, X_train_scaled.shape[1]]))]
        onnx_model = convert_sklearn(model, initial_types=initial_type)
        
        # Log the ONNX model to MLFlow registry too!
        mlflow.onnx.log_model(onnx_model, "onnx_model")
        
        onnx_path = os.path.join(models_dir, "yield_model.onnx")
        with open(onnx_path, "wb") as f:
            f.write(onnx_model.SerializeToString())
            
        print(f"Saved ONNX model to {onnx_path}")
        print("Models successfully tracked inside MLFlow!")

if __name__ == "__main__":
    build_and_train()
