# camera_demo.py
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.efficientnet import preprocess_input

# -------------------------
# Config
IMG_SIZE = 224
MODEL_PATH = "best_efficientnetb0.keras"

CLASS_NAMES = ['glass', 'metal', 'organic', 'paper', 'plastic', 'trash']
CONFIDENCE_THRESHOLD = 0.7   # ngưỡng "không có rác"

# -------------------------
# Load model
print("Loading model...")
model = load_model(MODEL_PATH)
print("Model loaded!")

# -------------------------
# Open camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Không thể mở camera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # OpenCV BGR -> RGB
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Resize
    img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))

    # Preprocess đúng chuẩn EfficientNet
    img_array = img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Predict
    probs = model.predict(img_array, verbose=0)[0]
    max_prob = float(np.max(probs))
    class_idx = int(np.argmax(probs))

    # Quyết định label
    if max_prob < CONFIDENCE_THRESHOLD:
        label = "không có rác"
        color = (0, 0, 255)  # đỏ
    else:
        label = CLASS_NAMES[class_idx]
        color = (0, 255, 0)  # xanh

    # Hiển thị kết quả
    text = f"{label}: {max_prob*100:.1f}%"
    cv2.putText(frame, text, (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("Garbage Classification Camera", frame)

    # Nhấn q để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
