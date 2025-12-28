# preprocess.py
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Đường dẫn dataset chứa train/, val/, test/
DATASET_DIR = "../dataset"

# Kích thước ảnh resize trước khi đưa vào model
IMG_SIZE = (224, 224)

# Số lượng ảnh mỗi batch
BATCH_SIZE = 32


def get_data_generators():
    """
    Tạo và trả về 3 generator: train, val, test.
    Train dùng data augmentation để cải thiện khả năng tổng quát hóa.
    Val/test dùng preprocessing đơn giản (rescale) để đánh giá đúng hơn.
    """

    # -----------------------------
    # 1) Data augmentation cho tập train
    # -----------------------------
    # Các phép biến đổi giúp tăng đa dạng dữ liệu và giảm overfitting:
    # - rotation_range: xoay ảnh trong khoảng ±15 độ
    # - width_shift_range / height_shift_range: tịnh tiến ảnh theo chiều ngang/dọc
    # - shear_range: biến dạng (shear transformation)
    # - zoom_range: phóng to/thu nhỏ ngẫu nhiên
    # - horizontal_flip: lật ngang ảnh
    # - fill_mode='nearest': các pixel ngoài vùng transform sẽ được điền bằng pixel gần nhất
    #
    # Tất cả ảnh đều được chuẩn hóa về [0,1] bằng rescale=1./255
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # -----------------------------
    # 2) Validation generator
    # -----------------------------
    # Validation KHÔNG sử dụng augmentation,
    # chỉ scale ảnh về [0,1] để đánh giá đúng khả năng thực của model.
    val_datagen = ImageDataGenerator(rescale=1./255)

    # -----------------------------
    # 3) Test generator
    # -----------------------------
    # Test cũng chỉ rescale để đảm bảo kết quả đánh giá chính xác.
    # shuffle=False để giữ nguyên thứ tự ảnh → cần thiết khi tính confusion matrix,
    # classification_report hoặc so khớp dự đoán theo file.
    test_datagen = ImageDataGenerator(rescale=1./255)

    # -----------------------------
    # Tạo generator từ thư mục ảnh
    # -----------------------------
    # flow_from_directory tự động:
    # - đọc ảnh từ folder theo cấu trúc class/
    # - sinh batch ảnh + nhãn
    # - resize ảnh về target_size
    # - class_mode='categorical' → one-hot label cho bài toán multi-class
    train_gen = train_datagen.flow_from_directory(
        os.path.join(DATASET_DIR, "train"),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    val_gen = val_datagen.flow_from_directory(
        os.path.join(DATASET_DIR, "val"),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    test_gen = test_datagen.flow_from_directory(
        os.path.join(DATASET_DIR, "test"),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False  # cần thiết để giữ thứ tự ảnh khi đánh giá
    )

    return train_gen, val_gen, test_gen


if __name__ == "__main__":
    train_gen, val_gen, test_gen = get_data_generators()
    print("Data generators ready!")
