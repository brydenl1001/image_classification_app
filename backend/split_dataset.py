import os
import random
import shutil

RANDOM_SEED = 42
SOURCE_DIR = os.path.join("raw_dataset", "Rice_Image_Dataset")
TARGET_DIR = "dataset"

SPLIT = {
    "train": 0.7,
    "val": 0.15,
    "test": 0.15
}

random.seed(RANDOM_SEED)

# Create target directories
for split in SPLIT:
    os.makedirs(os.path.join(TARGET_DIR, split), exist_ok=True)

for class_name in os.listdir(SOURCE_DIR):
    class_path = os.path.join(SOURCE_DIR, class_name)
    if not os.path.isdir(class_path):
        continue

    images = os.listdir(class_path)
    random.shuffle(images)

    n_total = len(images)
    n_train = int(n_total * SPLIT["train"])
    n_val = int(n_total * SPLIT["val"])

    splits = {
        "train": images[:n_train],
        "val": images[n_train:n_train + n_val],
        "test": images[n_train + n_val:]
    }

    for split, split_images in splits.items():
        split_class_dir = os.path.join(TARGET_DIR, split, class_name)
        os.makedirs(split_class_dir, exist_ok=True)

        for img in split_images:
            src = os.path.join(class_path, img)
            dst = os.path.join(split_class_dir, img)
            shutil.copy2(src, dst)

print("Dataset split complete.")
