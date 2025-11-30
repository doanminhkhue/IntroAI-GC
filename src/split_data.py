# split_data.py
import os
import shutil
import random

# Thư mục gốc chứa tất cả ảnh theo từng class
SOURCE_DIR = "data"  # folder ban đầu: dataset_raw/glass, dataset_raw/metal, ...
# Thư mục đích
DEST_DIR = "dataset"
# Tỷ lệ chia

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Danh sách class
CLASSES = ["glass", "metal", "organic", "paper", "plastic", "trash"]

def make_dirs():
    """Tạo thư mục train/val/test/class"""
    for split in ["train", "val", "test"]:
        for cls in CLASSES:
            dir_path = os.path.join(DEST_DIR, split, cls)
            os.makedirs(dir_path, exist_ok=True)

def split_data():
    make_dirs()
    for cls in CLASSES:
        src_cls_dir = os.path.join(SOURCE_DIR, cls)
        images = [f for f in os.listdir(src_cls_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        random.shuffle(images)
        n_total = len(images)
        n_train = int(n_total * TRAIN_RATIO)
        n_val = int(n_total * VAL_RATIO)
        n_test = n_total - n_train - n_val

        train_imgs = images[:n_train]
        val_imgs = images[n_train:n_train+n_val]
        test_imgs = images[n_train+n_val:]

        # Copy ảnh vào folder tương ứng
        for img in train_imgs:
            shutil.copy(os.path.join(src_cls_dir, img), os.path.join(DEST_DIR, "train", cls, img))
        for img in val_imgs:
            shutil.copy(os.path.join(src_cls_dir, img), os.path.join(DEST_DIR, "val", cls, img))
        for img in test_imgs:
            shutil.copy(os.path.join(src_cls_dir, img), os.path.join(DEST_DIR, "test", cls, img))

        print(f"[{cls}] Total: {n_total}, Train: {len(train_imgs)}, Val: {len(val_imgs)}, Test: {len(test_imgs)}")

if __name__ == "__main__":
    split_data()
    print("Done! Dataset split into train/val/test.")
