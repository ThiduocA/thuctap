import cv2
import pytesseract

#Đọc chính xác tiếng việt với ảnh sắc nét chưa qua chỉnh sửa
from PIL import Image
img_path = 'test1.jpg'
print(pytesseract.image_to_string(Image.open(img_path), lang='vie'))



