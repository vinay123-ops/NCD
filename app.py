from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import joblib
import numpy as np
from pydantic import BaseModel

# Load assets immediately
try:
    model = tf.keras.models.load_model('ncd_multilabel_risk_model.h5')
    scaler = joblib.load('scaler.pkl')
    ready = True
except:
    ready = False

app = FastAPI()

# Enable CORS for your React/Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class PatientData(BaseModel):
    HighChol: float; CholCheck: float; BMI: float; Smoker: float; Stroke: float
    PhysActivity: float; Fruits: float; Veggies: float; HvyAlcoholConsump: float
    AnyHealthcare: float; NoDocbcCost: float; GenHlth: float; MentHlth: float
    PhysHlth: float; DiffWalk: float; Sex: float; Age: float; Education: float; Income: float

@app.post("/predict")
def predict(data: PatientData):
    if not ready: raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Convert input to array in the exact order the model expects
    input_arr = np.array([[v for k, v in data.dict().items()]])
    scaled_input = scaler.transform(input_arr)
    prediction = model.predict(scaled_input, verbose=0)[0]
    
    return {
        "diabetes": f"{prediction[0]*100:.1f}%",
        "hypertension": f"{prediction[1]*100:.1f}%",
        "heart_disease": f"{prediction[2]*100:.1f}%"
    }