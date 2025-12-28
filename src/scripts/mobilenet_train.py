# train.py
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from preprocess import get_data_generators

# -----------------------------
# Cấu hình
# -----------------------------
IMG_SIZE = (224, 224)   # kích thước ảnh đầu vào cho model
NUM_CLASSES = 6         # số class trong dataset
BATCH_SIZE = 32
EPOCHS = 30
MODEL_NAME = "mobilenetv2"  # có thể thay bằng 'efficientnetb0'


def build_model(model_name="mobilenetv2"):
    """
    Xây dựng mô hình phân loại sử dụng Transfer Learning.
    Có thể chọn MobileNetV2 hoặc EfficientNetB0.
    """
    # -----------------------------
    # 1) Chọn base model pre-trained
    # include_top=False → bỏ fully connected layer cuối, chỉ giữ convolutional layers
    if model_name.lower() == "mobilenetv2":
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(*IMG_SIZE,3))
    else:
        base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(*IMG_SIZE,3))

    # -----------------------------
    # 2) Freeze base model
    # Giữ nguyên trọng số pre-trained, chỉ train phần FC mới để tránh overfitting và tiết kiệm thời gian
    base_model.trainable = False

    # -----------------------------
    # 3) Thêm custom head
    # GlobalAveragePooling2D: thay vì flatten → giảm số lượng tham số, tổng hợp đặc trưng
    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    # Dropout: regularization, giảm overfitting
    x = Dropout(0.3)(x)

    # Dense output layer với softmax cho multi-class classification
    predictions = Dense(NUM_CLASSES, activation='softmax')(x)

    # Kết hợp thành Model cuối cùng
    model = Model(inputs=base_model.input, outputs=predictions)
    return model


if __name__ == "__main__":
    # -----------------------------
    # 1) Load dữ liệu
    # -----------------------------
    train_gen, val_gen, test_gen = get_data_generators()

    # -----------------------------
    # 2) Build và compile model
    # -----------------------------
    model = build_model(MODEL_NAME)
    model.compile(
        optimizer=Adam(1e-4),                  # learning rate thấp để fine-tune head
        loss='categorical_crossentropy',       # multi-class classification
        metrics=['accuracy']
    )

    # -----------------------------
    # 3) Callbacks
    # -----------------------------
    # ModelCheckpoint: lưu model tốt nhất theo val_accuracy
    checkpoint = ModelCheckpoint("model/model.h5", monitor='val_accuracy', save_best_only=True, verbose=1)

    # EarlyStopping: dừng train nếu val_loss không cải thiện trong 5 epoch
    # restore_best_weights=True → model cuối cùng là model tốt nhất
    earlystop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)

    # -----------------------------
    # 4) Train model
    # -----------------------------
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=[checkpoint, earlystop]
    )

    print("Training finished!")

    # -----------------------------
    # 5) Vẽ đồ thị Accuracy và Loss
    # -----------------------------
    import matplotlib.pyplot as plt

    # Vẽ Accuracy
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig("accuracy_plot.png")
    plt.show()

    # Vẽ Loss
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig("loss_plot.png")
    plt.show()
