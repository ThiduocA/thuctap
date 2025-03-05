import cv2
import numpy as np
import pytesseract
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import gridspec
import os

def display_images(original, processed_images, figsize=(15, 5)):
    """
    Hiển thị ảnh gốc và các ảnh đã được xử lý
    """
    n = len(processed_images) + 1  # Số ảnh tổng cộng (gốc + xử lý)
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(1, n)
    
    # Hiển thị ảnh gốc
    ax = plt.subplot(gs[0])
    ax.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    ax.set_title("Original Image")
    ax.axis('off')
    
    # Hiển thị các ảnh đã xử lý
    for i, (name, img) in enumerate(processed_images):
        ax = plt.subplot(gs[i + 1])
        ax.imshow(img, cmap='gray')
        ax.set_title(name)
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

def enhance_image_for_ocr(image):
    """
    Áp dụng ba kỹ thuật xử lý ảnh: Adaptive Gaussian, Adaptive Mean, và Gaussian Blur + Adaptive
    """
    # Chuyển đổi ảnh sang grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 1. Adaptive Gaussian
    adaptive_gaussian = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                             cv2.THRESH_BINARY, 11, 2)
    
    # 2. Adaptive Mean
    adaptive_mean = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                         cv2.THRESH_BINARY, 11, 2)
    
    # 3. Gaussian Blur + Adaptive
    gaussian_blur = cv2.GaussianBlur(gray, (3, 3), 0)
    gaussian_adaptive = cv2.adaptiveThreshold(gaussian_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                             cv2.THRESH_BINARY, 11, 2)
    
    # Danh sách các ảnh đã xử lý
    processed_images = [
        ("Adaptive Gaussian", adaptive_gaussian),
        ("Adaptive Mean", adaptive_mean),
        ("Gaussian Blur + Adaptive", gaussian_adaptive)
    ]
    
    return processed_images

def recognize_text(image_path):
    """
    Nhận diện text từ ảnh sau khi áp dụng ba kỹ thuật xử lý
    """
    # Đọc ảnh
    original_image = cv2.imread(image_path)
    if original_image is None:
        print("Không thể đọc ảnh từ đường dẫn đã cho")
        return
    
    # Xử lý ảnh
    processed_images = enhance_image_for_ocr(original_image)
    
    # Hiển thị ảnh
    display_images(original_image, processed_images)
    
    # Nhận diện text từ mỗi ảnh đã xử lý
    for name, img in processed_images:
        # Lưu ảnh vào file tạm
        temp_filename = "temp.png"
        cv2.imwrite(temp_filename, img)
        
        # Nhận diện text bằng pytesseract
        text = pytesseract.image_to_string(Image.open(temp_filename), lang='vie')
        print(f"Text from {name}:\n{text}\n")
        
        # Xóa file tạm
        os.remove(temp_filename)

# Ví dụ sử dụng
if __name__ == "__main__":
    image_path = "test.png"  # Thay đổi đường dẫn ảnh tại đây
    recognize_text(image_path)