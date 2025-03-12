from fastapi import FastAPI, UploadFile, File, Form, HTTPException
import json
import numpy as np
import cv2
import insightface
import uvicorn

app = FastAPI()

# Khởi tạo model insightface
face_analysis = insightface.app.FaceAnalysis()
face_analysis.prepare(ctx_id=0)  # Sử dụng GPU nếu có, nếu không có thì dùng -1

def read_imagefile(file_bytes: bytes):
    """
    Chuyển bytes thành ảnh numpy dùng cv2.imdecode.
    Nếu không decode được, trả về None.
    """
    np_arr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return img

def get_face_embedding_from_array(img):
    """
    Dùng model insightface để phát hiện khuôn mặt và trích xuất embedding từ ảnh (numpy array).
    Nếu không phát hiện được khuôn mặt, trả về None.
    """
    try:
        faces = face_analysis.get(img)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi xử lý ảnh: {e}")
    if not faces or len(faces) == 0:
        return None
    # Giả sử chỉ xử lý khuôn mặt đầu tiên phát hiện
    face = faces[0]
    return face.embedding

def load_user_encodings(json_path):
    """
    Đọc file JSON chứa embedding và chuyển đổi mỗi embedding thành numpy array.
    Dữ liệu được lưu theo cấu trúc:
    {
        "user_id": {
            "front": [...],
            "left": [...],
            "right": [...]
        },
        ...
    }
    Nếu file không tồn tại hoặc lỗi decode, trả về dictionary rỗng.
    """
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        data = {}
    for user_id, angles in data.items():
        for angle in ["front", "left", "right"]:
            if angle in angles:
                angles[angle] = np.array(angles[angle])
    return data

def cosine_similarity(a, b):
    """
    Tính cosine similarity giữa hai vector a và b.
    Giá trị trả về nằm trong khoảng [-1, 1] (1 nghĩa là giống hệt).
    """
    try:
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi tính toán cosine similarity: {e}")

@app.post("/identify")
async def identify_face_endpoint(
    file_front: UploadFile = File(...),
    file_left: UploadFile = File(...),
    file_right: UploadFile = File(...),
    threshold: float = 0.5
):
    """
    Nhận diện khuôn mặt dựa trên 3 góc độ: front, left, right.
    Tính trung bình cosine similarity của 3 góc, nếu vượt qua threshold thì trả về user_id.
    """
    try:
        # Xử lý ảnh góc front
        bytes_front = await file_front.read()
        img_front = read_imagefile(bytes_front)
        if img_front is None:
            raise HTTPException(status_code=400, detail="File ảnh front không hợp lệ hoặc không thể đọc được.")
        embedding_front = get_face_embedding_from_array(img_front)
        if embedding_front is None:
            raise HTTPException(status_code=400, detail="Không phát hiện được khuôn mặt trong ảnh front.")

        # Xử lý ảnh góc left
        bytes_left = await file_left.read()
        img_left = read_imagefile(bytes_left)
        if img_left is None:
            raise HTTPException(status_code=400, detail="File ảnh left không hợp lệ hoặc không thể đọc được.")
        embedding_left = get_face_embedding_from_array(img_left)
        if embedding_left is None:
            raise HTTPException(status_code=400, detail="Không phát hiện được khuôn mặt trong ảnh left.")

        # Xử lý ảnh góc right
        bytes_right = await file_right.read()
        img_right = read_imagefile(bytes_right)
        if img_right is None:
            raise HTTPException(status_code=400, detail="File ảnh right không hợp lệ hoặc không thể đọc được.")
        embedding_right = get_face_embedding_from_array(img_right)
        if embedding_right is None:
            raise HTTPException(status_code=400, detail="Không phát hiện được khuôn mặt trong ảnh right.")

        json_path = "faces.json"
        user_encodings = load_user_encodings(json_path)
        if not user_encodings:
            raise HTTPException(status_code=404, detail="Chưa có người dùng nào được lưu trong hệ thống.")

        best_user = None
        best_avg_score = -1
        # Duyệt qua từng user và tính trung bình cosine similarity cho 3 góc
        for user_id, angles in user_encodings.items():
            if "front" not in angles or "left" not in angles or "right" not in angles:
                continue  # Bỏ qua user không có đủ thông tin 3 góc
            sim_front = cosine_similarity(embedding_front, angles["front"])
            sim_left = cosine_similarity(embedding_left, angles["left"])
            sim_right = cosine_similarity(embedding_right, angles["right"])
            avg_score = (sim_front + sim_left + sim_right) / 3.0
            if avg_score > best_avg_score:
                best_avg_score = avg_score
                best_user = user_id

        if best_avg_score > threshold:
            return {"user": best_user, "average_score": best_avg_score}
        else:
            return {"message": "Không nhận dạng được người dùng", "average_score": best_avg_score}
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/add_face")
async def add_face_endpoint(
    user_id: str = Form(...),
    file_front: UploadFile = File(...),
    file_left: UploadFile = File(...),
    file_right: UploadFile = File(...)
):
    """
    Thêm khuôn mặt của người dùng với 3 góc độ: front, left, right.
    Lưu embedding của từng góc vào file JSON.
    """
    try:
        # Xử lý ảnh góc front
        bytes_front = await file_front.read()
        img_front = read_imagefile(bytes_front)
        if img_front is None:
            raise HTTPException(status_code=400, detail="File ảnh front không hợp lệ hoặc không thể đọc được.")
        embedding_front = get_face_embedding_from_array(img_front)
        if embedding_front is None:
            raise HTTPException(status_code=400, detail="Không phát hiện được khuôn mặt trong ảnh front.")

        # Xử lý ảnh góc left
        bytes_left = await file_left.read()
        img_left = read_imagefile(bytes_left)
        if img_left is None:
            raise HTTPException(status_code=400, detail="File ảnh left không hợp lệ hoặc không thể đọc được.")
        embedding_left = get_face_embedding_from_array(img_left)
        if embedding_left is None:
            raise HTTPException(status_code=400, detail="Không phát hiện được khuôn mặt trong ảnh left.")

        # Xử lý ảnh góc right
        bytes_right = await file_right.read()
        img_right = read_imagefile(bytes_right)
        if img_right is None:
            raise HTTPException(status_code=400, detail="File ảnh right không hợp lệ hoặc không thể đọc được.")
        embedding_right = get_face_embedding_from_array(img_right)
        if embedding_right is None:
            raise HTTPException(status_code=400, detail="Không phát hiện được khuôn mặt trong ảnh right.")

        json_path = "faces.json"
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            data = {}

        data[user_id] = {
            "front": embedding_front.tolist(),
            "left": embedding_left.tolist(),
            "right": embedding_right.tolist()
        }
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=4)

        return {"message": f"Đã thêm khuôn mặt của {user_id} với 3 góc độ."}
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/identify_single")
async def identify_single_endpoint(
    angle: str = Form(...),
    file: UploadFile = File(...),
    threshold: float = Form(0.5)
):
    """
    Nhận diện khuôn mặt theo 1 góc độ cụ thể (front, left hoặc right).
    So sánh embedding của ảnh với embedding tương ứng của từng user trong file JSON.
    """
    # Kiểm tra góc nhận diện hợp lệ
    if angle not in ["front", "left", "right"]:
        raise HTTPException(status_code=400, detail="Angle không hợp lệ. Chọn một trong: front, left, right.")

    try:
        file_bytes = await file.read()
        img = read_imagefile(file_bytes)
        if img is None:
            raise HTTPException(status_code=400, detail="File ảnh không hợp lệ hoặc không thể đọc được.")
        embedding = get_face_embedding_from_array(img)
        if embedding is None:
            raise HTTPException(status_code=400, detail=f"Không phát hiện được khuôn mặt trong ảnh theo góc {angle}.")

        json_path = "faces.json"
        user_encodings = load_user_encodings(json_path)
        if not user_encodings:
            raise HTTPException(status_code=404, detail="Chưa có người dùng nào được lưu trong hệ thống.")

        best_user = None
        best_score = -1
        # So sánh embedding theo góc angle cho từng user
        for user_id, angles in user_encodings.items():
            if angle not in angles:
                continue
            score = cosine_similarity(embedding, angles[angle])
            if score > best_score:
                best_score = score
                best_user = user_id

        if best_score > threshold:
            return {"user": best_user, "score": best_score, "angle": angle}
        else:
            return {"message": "Không nhận dạng được người dùng", "score": best_score, "angle": angle}
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
