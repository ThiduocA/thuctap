import cv2
import os
import time

def capture_face_images(user_name):
    # Xử lý tên người dùng: thay khoảng trắng bằng dấu gạch dưới
    user_name = user_name.strip().replace(" ", "_")

    # Tạo folder với tên người dùng nếu chưa tồn tại
    folder_path = os.path.join(os.getcwd(), user_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Đã tạo folder: {folder_path}")

    # Khởi tạo camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Không thể truy cập camera!")
        return

    # Bộ nhận diện khuôn mặt Haar Cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    # Danh sách các góc chụp
    angles = [
        ("front", "Nhin thang vao camera"),
        ("right", "Quay dau sang phai"),
        ("left", "Quay dau sang trai")
    ]

    for angle_name, instruction in angles:
        image_path = os.path.join(folder_path, f"{angle_name}.jpg")
        
        # Kiểm tra nếu ảnh đã tồn tại thì bỏ qua
        if os.path.exists(image_path):
            print(f"Ảnh {angle_name} đã tồn tại, bỏ qua!")
            continue

        print(f"{instruction}")
        captured = False
        start_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            # Lật ảnh để giống gương trên điện thoại
            frame = cv2.flip(frame, 1)
            # Lưu lại bản gốc của frame trước khi vẽ overlay (để lưu ảnh sạch)
            frame_save = frame.copy()

            # Chuyển ảnh sang grayscale để phát hiện khuôn mặt
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

            # Vẽ khung nhận diện khuôn mặt (overlay hiển thị, không lưu vào ảnh)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Thêm hướng dẫn trên màn hình (overlay)
            cv2.putText(frame, instruction, (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (255, 255, 255), 2, cv2.LINE_AA)

            # Vẽ nút ảo giống như app camera trên điện thoại (overlay)
            height, width, _ = frame.shape
            button_center = (width // 2, height - 60)
            cv2.circle(frame, button_center, 40, (255, 255, 255), -1)
            cv2.circle(frame, button_center, 35, (0, 0, 0), 2)

            # Hiển thị khung camera toàn màn hình
            cv2.imshow("Camera", frame)

            # Nhấn 'c' để chụp hoặc tự động sau 5 giây (sử dụng frame_save để lưu ảnh không overlay)
            if cv2.waitKey(1) & 0xFF == ord('c') or (time.time() - start_time > 5 and not captured):
                cv2.imwrite(image_path, frame_save)
                print(f"Đã lưu ảnh {angle_name} tại {image_path}")
                captured = True
                time.sleep(1)
                break

    # Giải phóng tài nguyên
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    user_name = input("Nhập tên người dùng: ")
    capture_face_images(user_name)
