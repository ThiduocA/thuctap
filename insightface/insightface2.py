import json
import numpy as np
import cv2
import insightface

# Khởi tạo và chuẩn bị model insightface
app = insightface.app.FaceAnalysis()
app.prepare(ctx_id=0)  # ctx_id=0 dùng GPU nếu có, nếu không có GPU thì dùng -1

def get_face_embedding(img_path):
    """
    Hàm này đọc ảnh từ img_path, phát hiện khuôn mặt và trích xuất embedding.
    Nếu không phát hiện được khuôn mặt, trả về None.
    """
    img = cv2.imread(img_path)
    faces = app.get(img)
    if len(faces) == 0:
        print("Không phát hiện được khuôn mặt trong ảnh!")
        return None
    # Giả sử chỉ xử lý khuôn mặt đầu tiên phát hiện
    face = faces[0]
    return face.embedding

def load_user_encodings(json_path):
    """
    Hàm này đọc file JSON chứa các embedding của người dùng và chuyển đổi mỗi embedding thành numpy array.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    # Chuyển đổi mỗi embedding thành numpy array để thuận tiện cho tính toán
    for user_id, vec in data.items():
        data[user_id] = np.array(vec)
    return data

def cosine_similarity(a, b):
    """
    Tính cosine similarity giữa hai vector a và b.
    Giá trị trả về nằm trong khoảng [-1, 1] (1 nghĩa là giống hệt).
    """
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def identify_face(image_path, json_path, threshold=0.5):
    """
    Hàm nhận đầu vào là đường dẫn tới ảnh và file JSON chứa embedding.
    - Trích xuất embedding của khuôn mặt trong ảnh.
    - So sánh với từng embedding trong file JSON sử dụng cosine similarity.
    - Nếu similarity cao hơn threshold, trả về user_id và điểm similarity.
    
    Tham số threshold cần được điều chỉnh dựa trên thực nghiệm và chất lượng của model.
    """
    embedding = get_face_embedding(image_path)
    if embedding is None:
        return None, None
    
    user_encodings = load_user_encodings(json_path)
    
    best_user = None
    best_score = -1
    for user_id, user_vec in user_encodings.items():
        score = cosine_similarity(embedding, user_vec)
        if score > best_score:
            best_score = score
            best_user = user_id
            
    if best_score > threshold:
        return best_user, best_score
    else:
        return None, best_score

def add_face_to_json(user_id, image_path, json_path):
    """
    Hàm thêm khuôn mặt của người dùng vào file JSON chứa embedding.

    Tham số:
    - user_id: Mã định danh của người dùng (string)
    - image_path: Đường dẫn tới ảnh của người dùng cần thêm
    - json_path: Đường dẫn tới file JSON chứa embedding

    Quy trình:
    1. Sử dụng hàm get_face_embedding để trích xuất embedding từ ảnh.
    2. Đọc file JSON hiện có, nếu file không tồn tại thì khởi tạo dictionary rỗng.
    3. Chuyển embedding (numpy array) sang list để lưu trữ dưới định dạng JSON.
    4. Ghi lại file JSON với dữ liệu đã cập nhật.
    """
    embedding = get_face_embedding(image_path)
    if embedding is None:
        print("Không nhận dạng được khuôn mặt trong ảnh!")
        return False

    # Đọc file JSON nếu có, ngược lại khởi tạo dictionary rỗng
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        data = {}

    # Thêm hoặc cập nhật embedding của user_id (chuyển numpy array thành list)
    data[user_id] = embedding.tolist()

    # Ghi lại dữ liệu vào file JSON
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=4)

    print(f"Đã thêm khuôn mặt của {user_id} vào file JSON.")
    return True

# Ví dụ sử dụng
if __name__ == "__main__":
    #add_face_to_json("0012xxxx4047", "XuanHung.jpg", "faces.json")
    image_path = "cccd.jpg"    # đường dẫn tới ảnh cần xác minh
    json_path = "faces.json"   # đường dẫn tới file JSON chứa embedding
    user, score = identify_face(image_path, json_path, threshold=0.5)
    # embedding = get_face_embedding("cccd.jpg")
    # print(embedding)
    if user:
        print(f"Nhận dạng thành công: {user} với điểm tương đồng: {score:.2f}")
    else:
        print(f"Không nhận dạng được người dùng (điểm cao nhất: {score:.2f})")
    
