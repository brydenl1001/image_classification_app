import os
import random
import shutil
from pathlib import Path

# ==========================
# CONFIG
# ==========================
SOURCE_DIR = os.path.join("raw_waste_dataset", "images", "images")
OUTPUT_DIR = "waste_dataset"

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")

random.seed(42)

# ==========================
# CREATE OUTPUT DIRS
# ==========================
for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(OUTPUT_DIR, split), exist_ok=True)

# ==========================
# PROCESS EACH CLASS
# ==========================
for class_name in os.listdir(SOURCE_DIR):
    class_path = os.path.join(SOURCE_DIR, class_name)
    if not os.path.isdir(class_path):
        continue

    images = []

    # Collect images from BOTH subfolders
    for subfolder in ["default", "real_world"]:
        sub_path = os.path.join(class_path, subfolder)
        if not os.path.exists(sub_path):
            continue

        for img in os.listdir(sub_path):
            if img.lower().endswith(IMAGE_EXTENSIONS):
                images.append(os.path.join(sub_path, img))

    if len(images) == 0:
        print(f"⚠️ No images found for class {class_name}")
        continue

    random.shuffle(images)

    n_total = len(images)
    n_train = int(n_total * TRAIN_RATIO)
    n_val = int(n_total * VAL_RATIO)

    splits = {
        "train": images[:n_train],
        "val": images[n_train:n_train + n_val],
        "test": images[n_train + n_val:]
    }

    # Create class folders and copy images
    for split, split_images in splits.items():
        split_class_dir = os.path.join(OUTPUT_DIR, split, class_name)
        os.makedirs(split_class_dir, exist_ok=True)

        for img_path in split_images:
            dst = os.path.join(split_class_dir, Path(img_path).name)
            shutil.copy(img_path, dst)

    print(f"✅ {class_name}: {n_total} images split")

print("🎉 Dataset split complete.")
