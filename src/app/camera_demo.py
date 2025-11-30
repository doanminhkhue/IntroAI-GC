# camera_demo.py
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# -----------------------------
# Cấu hình
# -----------------------------
IMG_SIZE = 224  # kích thước ảnh đầu vào cho model
MODEL_PATH = "../scripts/model/model.h5"

# -----------------------------
# 1) Load model
# -----------------------------
# Model đã train trước đó
model = load_model(MODEL_PATH)

# Danh sách class tương ứng index output của model
class_names = ['glass', 'metal', 'organic', 'paper', 'plastic', 'trash']

# -----------------------------
# 2) Mở camera
# -----------------------------
# 0 → camera mặc định của máy
cap = cv2.VideoCapture(0)

# -----------------------------
# 3) Vòng lặp đọc camera liên tục
# -----------------------------
while True:
    ret, frame = cap.read()  # ret=True nếu frame đọc thành công
    if not ret:
        break

    # -----------------------------
    # 4) Preprocess frame
    # -----------------------------
    # Resize về IMG_SIZE x IMG_SIZE
    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))

    # Chuyển sang mảng numpy, scale về [0,1]
    img = img_to_array(img) / 255.0

    # Thêm batch dimension: model cần input shape (1, IMG_SIZE, IMG_SIZE, 3)
    img = np.expand_dims(img, axis=0)

    # -----------------------------
    # 5) Predict
    # -----------------------------
    pred = model.predict(img)

    # Chọn class có xác suất cao nhất
    class_idx = np.argmax(pred)
    label = class_names[class_idx]
    confidence = pred[0][class_idx]  # xác suất dự đoán class này

    # -----------------------------
    # 6) Hiển thị kết quả lên frame
    # -----------------------------
    cv2.putText(frame, f"{label}: {confidence*100:.1f}%", 
                (10,30),  # vị trí text
                cv2.FONT_HERSHEY_SIMPLEX, 
                1,       # font scale
                (0,255,0), # màu xanh lá
                2)       # độ dày line

    # Hiển thị cửa sổ camera
    cv2.imshow("Camera Demo", frame)

    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# -----------------------------
# 7) Giải phóng tài nguyên
# -----------------------------
cap.release()
cv2.destroyAllWindows()
