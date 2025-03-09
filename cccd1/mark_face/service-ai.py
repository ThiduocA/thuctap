# Macos 
# brew install cmake
# brew install openblas
# Ubuntu 
# sudo apt-get update
# sudo apt-get install build-essential cmake
# sudo apt-get install libopenblas-dev liblapack-dev
# sudo apt-get install libx11-dev libgtk-3-dev
# pip3 install fastapi uvicorn face_recognition numpy opencv-python


import face_recognition
import numpy as np
import cv2
import os
import json

from fastapi import FastAPI, UploadFile, File

app = FastAPI()

FACE_DB = "faces.json"  # Lưu trữ encoding khuôn mặt

def load_faces():
    """Tải dữ liệu khuôn mặt đã đăng ký từ file JSON"""
    if os.path.exists(FACE_DB):
        with open(FACE_DB, "r") as f:
            return json.load(f)
    return {}

def save_faces(data):
    """Lưu dữ liệu khuôn mặt vào file JSON"""
    with open(FACE_DB, "w") as f:
        json.dump(data, f)

faces = load_faces()

@app.post("/register_face/")
async def register_face(name: str, file: UploadFile = File(...)):
    """API đăng ký khuôn mặt mới"""
    image = face_recognition.load_image_file(file.file)
    encodings = face_recognition.face_encodings(image)

    if not encodings:
        return {"error": "Không tìm thấy khuôn mặt!"}

    faces[name] = encodings[0].tolist()  # Lưu encoding dưới dạng list
    save_faces(faces)

    return {"message": f"Khuôn mặt {name} đã được đăng ký thành công"}

@app.post("/recognize_face/")
async def recognize_face(file: UploadFile = File(...)):
    """API nhận diện khuôn mặt"""
    image = face_recognition.load_image_file(file.file)
    encodings = face_recognition.face_encodings(image)

    if not encodings:
        return {"error": "Không tìm thấy khuôn mặt!"}

    encoding_to_check = encodings[0]
    known_faces = {name: np.array(enc) for name, enc in faces.items()}

    for name, encoding in known_faces.items():
        match = face_recognition.compare_faces([encoding], encoding_to_check)
        if match[0]:
            return {"message": f"Nhận diện thành công: {name}"}

    return {"message": "Không tìm thấy khuôn mặt trùng khớp"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
