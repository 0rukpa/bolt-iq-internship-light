from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# 1. Initialize the FastAPI app instance
app = FastAPI()

# 2. Define the exact JSON schema your API expects to receive
class PredictionInput(BaseModel):
    rsi: float
    sma_diff: float
    pcr: float
    atr: float
    hlr: float

# 3. Load the model into memory ONCE from its directory path when the app starts



import os

# Find the directory where main.py is currently sitting
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Go up one folder level to the root, then look for the model file
# Try your project root first, fall back to your user profile directory if needed
MODEL_PATH = os.path.join(BASE_DIR, "..", "model", "model.pkl")

# If it's not in the project root, point it directly to your C:\Users\HP folder location
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join("C:\\", "Users", "HP", "model", "model.pkl")

model = joblib.load(MODEL_PATH)

# 4. Create your health check endpoint
@app.get("/health")
def health_check():
    return {"status": "OK"}

# 5. Create your prediction POST endpoint
@app.post("/predict")
def predict_direction(input_data: PredictionInput):
    # Collect features into a sequence list
    features_list = [
        input_data.rsi, 
        input_data.sma_diff, 
        input_data.pcr, 
        input_data.atr, 
        input_data.hlr
    ]
    
    # Reshape the list into a 2D array matrix for scikit-learn
    features_array = np.array(features_list).reshape(1, -1)
    
    # Generate the prediction and extract the raw integer out of the numpy wrapper
    raw_prediction = model.predict(features_array)
    final_result = int(raw_prediction[0])
    
    # Return the clean prediction dictionary across the web
    return {"prediction": final_result}
