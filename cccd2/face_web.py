from fastapi import FastAPI, UploadFile, File, HTTPException
import cv2
import os
import numpy as np
import face_recognition
import shutil
import time

app = FastAPI()

# Thư mục lưu ảnh người dùng
IMAGE_FOLDER = "./imgs/persons"
os.makedirs(IMAGE_FOLDER, exist_ok=True)

# Load dữ liệu nhận diện khuôn mặt
def load_known_faces():
    known_face_encodings = []
    known_face_names = []
    for person_folder in os.listdir(IMAGE_FOLDER):
        person_path = os.path.join(IMAGE_FOLDER, person_folder)
        if os.path.isdir(person_path):
            encodings = []
            for filename in os.listdir(person_path):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(person_path, filename)
                    image = face_recognition.load_image_file(image_path)
                    face_encodings = face_recognition.face_encodings(image)
                    if face_encodings:
                        encodings.append(face_encodings[0])
            if encodings:
                known_face_encodings.append(np.mean(encodings, axis=0))
                known_face_names.append(person_folder)
    return known_face_encodings, known_face_names

known_face_encodings, known_face_names = load_known_faces()

@app.get("/list_images/")
def list_images():
    if not os.path.exists(IMAGE_FOLDER):
        return {"error": f"Thư mục {IMAGE_FOLDER} không tồn tại."}
    
    data = {}
    for person_folder in os.listdir(IMAGE_FOLDER):
        person_path = os.path.join(IMAGE_FOLDER, person_folder)
        if os.path.isdir(person_path):
            images = [f for f in os.listdir(person_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            data[person_folder] = images
    
    return {"folders": data}

@app.post("/upload/")
async def upload_file(user_name: str, file: UploadFile = File(...)):
    user_folder = os.path.join(IMAGE_FOLDER, user_name)
    os.makedirs(user_folder, exist_ok=True)
    file_path = os.path.join(user_folder, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"message": f"File uploaded successfully: {file.filename}"}

# Các endpoint còn lại giữ nguyên
@app.post("/check/")
def check_face_live():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise HTTPException(status_code=500, detail="Could not open webcam")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        
        frame = cv2.flip(frame, 1)
        rgb_frame = frame[:, :, ::-1]
        face_encodings = face_recognition.face_encodings(rgb_frame)
        
        if face_encodings:
            face_encoding = face_encodings[0]
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"
            if True in matches:
                best_match_index = np.argmin(face_recognition.face_distance(known_face_encodings, face_encoding))
                name = known_face_names[best_match_index]
            
            cap.release()
            cv2.destroyAllWindows()
            return {"name": name}
        
        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    raise HTTPException(status_code=400, detail="No face detected")

@app.post("/capture/")
def capture_face_images(user_name: str):
    user_name = user_name.strip().replace(" ", "_")
    user_folder = os.path.join(IMAGE_FOLDER, user_name)
    os.makedirs(user_folder, exist_ok=True)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise HTTPException(status_code=500, detail="Could not open webcam")
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    angles = [
        ("front", "Nhìn thẳng vào camera"),
        ("right", "Quay đầu sang phải"),
        ("left", "Quay đầu sang trái")
    ]
    
    for angle_name, instruction in angles:
        image_path = os.path.join(user_folder, f"{angle_name}.jpg")
        if os.path.exists(image_path):
            continue
        
        print(f"{instruction}")
        captured = False
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            
            frame = cv2.flip(frame, 1)
            frame_save = frame.copy()
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
            
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            cv2.putText(frame, instruction, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            height, width, _ = frame.shape
            button_center = (width // 2, height - 60)
            cv2.circle(frame, button_center, 40, (255, 255, 255), -1)
            cv2.circle(frame, button_center, 35, (0, 0, 0), 2)
            
            cv2.imshow("Camera", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('c') or (time.time() - start_time > 5 and not captured):
                cv2.imwrite(image_path, frame_save)
                captured = True
                time.sleep(1)
                break
    
    cap.release()
    cv2.destroyAllWindows()
    return {"message": f"Images captured for {user_name}"}