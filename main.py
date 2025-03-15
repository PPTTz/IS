import numpy as np
import tensorflow as tf
import base64
import os
import io
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from tensorflow.keras.preprocessing import image
from PIL import Image
import joblib
import gdown
import re
from tensorflow.keras.models import load_model  # ✅ เพิ่มบรรทัดนี้
from tensorflow.keras.mixed_precision import Policy
from tensorflow.keras.initializers import GlorotUniform, Zeros

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# URLs สำหรับดาวน์โหลดโมเดล
url_KNN = "https://drive.google.com/uc?export=download&id=1p1LENJdRXt4OfkL803WZHv7-khrumBsO"
url_CNN = "https://drive.google.com/uc?export=download&id=1KhjmTRLFyxX0x6KYisgfBXQ4FnbIFkc0"
url_SVM = "https://drive.google.com/uc?export=download&id=1ArIE2t1LC07J2lYkOE3KMKYCJ6YL7Loy"


# ตรวจสอบและโหลดโมเดล KNN
knn_model_path = "knn_model.pkl"
if not os.path.exists(knn_model_path):
    gdown.download(url_KNN, knn_model_path, quiet=False)
knn_model = joblib.load(knn_model_path)

# ตรวจสอบและโหลดโมเดล SVM
svm_model_path = "svm_model.pkl"
if not os.path.exists(svm_model_path):
    gdown.download(url_SVM, svm_model_path, quiet=False)
svm_model = joblib.load(svm_model_path)

# ตรวจสอบและโหลดโมเดล CNN
DogVsCat_model_path = "DogVsCat.h5"
if not os.path.exists(DogVsCat_model_path):
    gdown.download(url_CNN, DogVsCat_model_path, quiet=False)

DogVsCat_model = load_model(DogVsCat_model_path, compile=False, safe_mode=False)

class_labels = [0, 1]

class DiabetesData(BaseModel):
    gender: int
    age: int
    hypertension: int
    heart_disease: int
    smoking_history: int
    bmi: int
    HbA1c_level: int
    blood_glucose_level: int

class ImageData(BaseModel):
    image_base64: str

# ทำนายภาวะหัวใจ (KNN)
@app.post("/predict/KNN") 
def predict_Disease_knn(data: DiabetesData):
    input_data = np.array([[data.gender, data.age, data.hypertension, data.heart_disease, data.smoking_history,
                            data.bmi, data.HbA1c_level, data.blood_glucose_level]])
    prediction = knn_model.predict(input_data)[0]
    return {"result": "High Risk" if prediction == 1 else "Low Risk"}

# ทำนายภาวะหัวใจ (SVM)
@app.post("/predict/SVM") 
def predict_Disease_svm(data: DiabetesData):
    input_data = np.array([[data.gender, data.age, data.hypertension, data.heart_disease, data.smoking_history,
                            data.bmi, data.HbA1c_level, data.blood_glucose_level]])
    prediction = svm_model.predict(input_data)[0]
    return {"result": "High Risk" if prediction == 1 else "Low Risk"}

# ทำนาย Dog vs Cat
@app.post("/predict/DogVsCat")
async def predict_DogVsCat(image_data: ImageData):
    try:
        base64_data = re.sub(r'^data:image/[^;]+;base64,', '', image_data.image_base64)
        image_bytes = base64.b64decode(base64_data)
        img = Image.open(io.BytesIO(image_bytes))
        
        if img.mode != "RGB":
            img = img.convert("RGB")
        
        img = img.resize((64, 64))  # ✅ แก้ไข resize ให้เป็น (64, 64)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  
        img_array = img_array / 127.5 - 1  

        predictions = DogVsCat_model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)[0]
        predicted_label = class_labels[predicted_class]

        return JSONResponse(content={"prediction": predicted_label})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
