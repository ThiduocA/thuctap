import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
import os
import logging

# Cấu hình logging để theo dõi thông tin debug
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_face_recognition_model(ctx_id=0, det_size=(640, 640)):
    app = FaceAnalysis(name='buffalo_l')
    app.prepare(ctx_id=ctx_id, det_size=det_size)
    logger.info("Model đã được tải với ctx_id=%s và det_size=%s", ctx_id, det_size)
    return app

def load_image(image_path):
    if not os.path.exists(image_path):
        logger.error("File không tồn tại: %s", image_path)
        return None
    img = cv2.imread(image_path)
    if img is None:
        logger.error("Không thể đọc file ảnh: %s", image_path)
    return img

def get_face_embedding(app, image_path):
    img = load_image(image_path)
    if img is None:
        return None

    faces = app.get(img)
    if faces and len(faces) > 0:
        logger.info("Phát hiện %d khuôn mặt trong ảnh: %s", len(faces), image_path)
        return faces[0].normed_embedding
    else:
        logger.warning("Không phát hiện khuôn mặt nào trong ảnh: %s", image_path)
    return None

def cosine_similarity(emb1, emb2):
    if emb1 is None or emb2 is None:
        return -1
    return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

def recognize_face(app, db_embeddings, db_labels, test_image_path, threshold=0.5):
    test_embedding = get_face_embedding(app, test_image_path)
    if test_embedding is None:
        return "No face detected", None

    best_match = None
    best_score = -1
    for label, db_embedding in zip(db_labels, db_embeddings):
        score = cosine_similarity(test_embedding, db_embedding)
        logger.debug("So sánh với %s: similarity = %.4f", label, score)
        if score > best_score:
            best_score = score
            best_match = label

    if best_score >= threshold:
        return best_match, best_score
    else:
        return "Unknown", best_score

if __name__ == "__main__":
    # Khởi tạo model
    app = load_face_recognition_model(ctx_id=0, det_size=(640, 640))

    # Đường dẫn tới thư mục chứa ảnh (sử dụng os.path.join để đảm bảo tính tương thích)
    img_folder_path = os.getcwd() + "/imgs"
    image1_path = os.path.join(img_folder_path, "front.jpg")
    image2_path = os.path.join(img_folder_path, "cccd.jpg")

    # Tạo database khuôn mặt từ các ảnh mẫu
    face_db = {
        "person1": get_face_embedding(app, image1_path),
        "person2": get_face_embedding(app, image2_path)
    }
    db_embeddings = list(face_db.values())
    db_labels = list(face_db.keys())

    # Kiểm tra nhận diện khuôn mặt với ảnh test (có thể thay đổi đường dẫn test nếu cần)
    test_image_path = image1_path
    result, confidence = recognize_face(app, db_embeddings, db_labels, test_image_path)
    if confidence is not None:
        print(f"Recognition result: {result} (Confidence: {confidence:.2f})")
    else:
        print("No face detected")
