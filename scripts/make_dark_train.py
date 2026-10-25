import cv2, numpy as np, os, shutil
from tqdm import tqdm

SRC = "dataset/trashcan_full"
OUT = "dataset/trashcan_full_darkaware"
MAX_LEVEL = 0.7          # training darkness range is 0 .. 0.7
KEEP_BRIGHT = 0.2        # fraction of images left unchanged
SEED = 123               # different from the test-set seed

# IMPORTANT: this function must be identical to the one in make_dark_sets.py
def darken(img, level, rng):
    x = img.astype(np.float32) / 255.0
    atten = np.array([1.0 - 0.4 * level, 1.0 - 0.7 * level, 1.0 - 0.95 * level], dtype=np.float32)
    x = x * atten
    x = np.power(x, 1.0 + 1.5 * level) * (1.0 - 0.6 * level)
    noise = rng.normal(0, 0.01 + 0.04 * level, x.shape).astype(np.float32)
    x = np.clip(x + noise, 0, 1)
    return (x * 255).astype(np.uint8)

rng = np.random.default_rng(SEED)
for split in ["train", "val"]:
    os.makedirs(f"{OUT}/images/{split}", exist_ok=True)
    os.makedirs(f"{OUT}/labels/{split}", exist_ok=True)
    for fname in tqdm(os.listdir(f"{SRC}/images/{split}"), desc=split):
        img = cv2.imread(f"{SRC}/images/{split}/{fname}")
        if img is None:
            continue
        if rng.random() > KEEP_BRIGHT:
            img = darken(img, rng.uniform(0.05, MAX_LEVEL), rng)
        cv2.imwrite(f"{OUT}/images/{split}/{fname}", img)
        stem = os.path.splitext(fname)[0]
        shutil.copy(f"{SRC}/labels/{split}/{stem}.txt", f"{OUT}/labels/{split}/{stem}.txt")
print("Done:", OUT)