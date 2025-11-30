# streamlit_app.py
import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import time

# -------------------------
# Config
IMG_SIZE = 224
MODEL_PATH = "C:/Users/ASUS/Documents/AI_Project/PJ2/scripts/model/model.h5"
CLASS_NAMES = ['glass', 'metal', 'organic', 'paper', 'plastic', 'trash']

# -------------------------
# Load model
@st.cache_resource
def load_trained_model():
    return load_model(MODEL_PATH)

model = load_trained_model()


# -------------------------
# Predict function
def predict_image(img):
    img_resized = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = img_to_array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    pred = model.predict(img_array, verbose=0)
    class_idx = np.argmax(pred)
    label = CLASS_NAMES[class_idx]
    confidence = pred[0][class_idx]
    return label, confidence


# -------------------------
# Streamlit UI
st.title("Garbage Classifier")
st.write("Ứng dụng phân loại rác bằng Camera hoặc Upload ảnh")

option = st.radio("Chọn chế độ:", ("Upload ảnh", "Camera real-time"))

# -------------------------
# UPLOAD ẢNH
if option == "Upload ảnh":
    uploaded_file = st.file_uploader("Chọn ảnh", type=["jpg","jpeg","png"])
    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Ảnh đã chọn", use_column_width=True)
        label, confidence = predict_image(img)
        st.success(f"Nhận diện: {label} ({confidence*100:.1f}%)")


# -------------------------
# CAMERA REALTIME (Phiên bản cải tiến)
elif option == "Camera real-time":

    st.subheader("Nhận diện qua Camera")

    # Tạo biến trạng thái nếu chưa có
    if "camera_running" not in st.session_state:
        st.session_state.camera_running = False

    # Nút START
    if st.button("Bắt đầu Camera"):
        st.session_state.camera_running = True

    # Nút STOP
    if st.button("Dừng Camera"):
        st.session_state.camera_running = False

    # Khung hiển thị camera
    frame_window = st.image([])

    if st.session_state.camera_running:
        cap = cv2.VideoCapture(0)

        while st.session_state.camera_running:

            ret, frame = cap.read()
            if not ret:
                st.error("Không thể mở camera")
                break

            # Chuyển về RGB và dự đoán
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            label, confidence = predict_image(img_pil)

            # Hiển thị label lên frame
            cv2.putText(frame, f"{label}: {confidence*100:.1f}%", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

            frame_window.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # Giảm tải CPU
            time.sleep(0.03)

        cap.release()
