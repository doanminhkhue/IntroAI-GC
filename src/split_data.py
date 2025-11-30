# split_data.py
import os
import shutil
import random

# Thư mục gốc chứa tất cả ảnh theo từng class
# Ví dụ: data/glass, data/metal, ...
SOURCE_DIR = "data"

# Thư mục đích sau khi chia
# Sẽ tạo cấu trúc như:
# dataset/train/glass
# dataset/val/glass
# dataset/test/glass
DEST_DIR = "dataset"

# Tỷ lệ chia dataset
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Danh sách class có trong dataset
CLASSES = ["glass", "metal", "organic", "paper", "plastic", "trash"]


def make_dirs():
    """
    Tạo các thư mục đích cho từng split (train/val/test)
    và cho từng class.
    
    Cấu trúc tạo ra:
        dataset/
            train/
                glass/
                metal/
                ...
            val/
                glass/
                metal/
                ...
            test/
                glass/
                metal/
                ...
    """
    for split in ["train", "val", "test"]:
        for cls in CLASSES:
            dir_path = os.path.join(DEST_DIR, split, cls)
            os.makedirs(dir_path, exist_ok=True)  # exist_ok=True → không lỗi nếu thư mục đã tồn tại


def split_data():
    make_dirs()  # Tạo thư mục trước khi bắt đầu copy

    for cls in CLASSES:
        # Đường dẫn đến folder chứa ảnh của class hiện tại
        src_cls_dir = os.path.join(SOURCE_DIR, cls)

        # Lấy danh sách ảnh (lọc theo phần mở rộng)
        images = [f for f in os.listdir(src_cls_dir)
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        # Trộn ngẫu nhiên thứ tự ảnh để tránh bias
        random.shuffle(images)

        # Tổng số ảnh của class
        n_total = len(images)

        # Tính số lượng theo tỷ lệ
        # int() để làm tròn xuống → giúp đảm bảo tổng không vượt quá số ảnh thực tế
        n_train = int(n_total * TRAIN_RATIO)
        n_val = int(n_total * VAL_RATIO)

        # Số ảnh test còn lại (đảm bảo tổng = n_total)
        n_test = n_total - n_train - n_val

        # Chia theo index
        # 0 → n_train: train
        train_imgs = images[:n_train]
        # n_train → n_train+n_val: val
        val_imgs = images[n_train:n_train + n_val]
        # còn lại là test
        test_imgs = images[n_train + n_val:]

        # Copy ảnh vào thư mục đích
        for img in train_imgs:
            shutil.copy(os.path.join(src_cls_dir, img),
                        os.path.join(DEST_DIR, "train", cls, img))

        for img in val_imgs:
            shutil.copy(os.path.join(src_cls_dir, img),
                        os.path.join(DEST_DIR, "val", cls, img))

        for img in test_imgs:
            shutil.copy(os.path.join(src_cls_dir, img),
                        os.path.join(DEST_DIR, "test", cls, img))

        # In thông tin để kiểm tra
        print(f"[{cls}] Total: {n_total}, Train: {len(train_imgs)}, Val: {len(val_imgs)}, Test: {len(test_imgs)}")


if __name__ == "__main__":
    split_data()
    print("Done! Dataset split into train/val/test.")
