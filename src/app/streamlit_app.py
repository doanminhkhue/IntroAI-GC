# streamlit_app.py
import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import time

# -------------------------
# 1) Config
IMG_SIZE = 224  # kích thước ảnh input cho model
MODEL_PATH = "C:/Users/ASUS/Documents/AI_Project/PJ2/scripts/model/model.h5"
CLASS_NAMES = ['glass', 'metal', 'organic', 'paper', 'plastic', 'trash']

# -------------------------
# 2) Load model với cache
@st.cache_resource
def load_trained_model():
    """
    Load model và cache để tránh load lại mỗi lần rerun Streamlit.
    st.cache_resource giúp giữ model trong bộ nhớ khi Streamlit reload script.
    """
    return load_model(MODEL_PATH)

model = load_trained_model()

# -------------------------
# 3) Hàm predict cho 1 ảnh
def predict_image(img):
    """
    img: PIL Image
    Trả về nhãn dự đoán và xác suất
    """
    # Resize về kích thước model yêu cầu
    img_resized = img.resize((IMG_SIZE, IMG_SIZE))

    # Chuyển sang numpy array và scale về [0,1]
    img_array = img_to_array(img_resized) / 255.0

    # Thêm batch dimension (model nhận input dạng (1, H, W, 3))
    img_array = np.expand_dims(img_array, axis=0)

    # Dự đoán
    pred = model.predict(img_array, verbose=0)

    # Chọn class có xác suất cao nhất
    class_idx = np.argmax(pred)
    label = CLASS_NAMES[class_idx]
    confidence = pred[0][class_idx]
    return label, confidence

# -------------------------
# 4) Streamlit UI
st.title("Garbage Classifier")
st.write("Ứng dụng phân loại rác bằng Camera hoặc Upload ảnh")

# Radio button để chọn chế độ: Upload ảnh hoặc Camera real-time
option = st.radio("Chọn chế độ:", ("Upload ảnh", "Camera real-time"))

# -------------------------
# 5) Upload ảnh
if option == "Upload ảnh":
    uploaded_file = st.file_uploader("Chọn ảnh", type=["jpg","jpeg","png"])
    if uploaded_file:
        # Chuyển sang PIL Image
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Ảnh đã chọn", use_column_width=True)

        # Dự đoán nhãn và xác suất
        label, confidence = predict_image(img)
        st.success(f"Nhận diện: {label} ({confidence*100:.1f}%)")

# -------------------------
# 6) Camera real-time
elif option == "Camera real-time":
    st.subheader("Nhận diện qua Camera")

    # Sử dụng session_state để lưu trạng thái camera
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
        cap = cv2.VideoCapture(0)  # mở camera mặc định

        while st.session_state.camera_running:

            # Đọc frame
            ret, frame = cap.read()
            if not ret:
                st.error("Không thể mở camera")
                break

            # Chuyển BGR → RGB vì OpenCV đọc frame dạng BGR, PIL dùng RGB
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)

            # Dự đoán nhãn
            label, confidence = predict_image(img_pil)

            # Hiển thị nhãn + confidence lên frame
            cv2.putText(frame, f"{label}: {confidence*100:.1f}%", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

            # Hiển thị frame trong Streamlit
            frame_window.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # Giảm tải CPU, tạo delay ~30ms
            time.sleep(0.03)

        cap.release()  # giải phóng camera khi dừng
