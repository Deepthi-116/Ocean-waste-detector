import cv2, numpy as np, os, shutil
from tqdm import tqdm

SRC = "dataset/trashcan_full"
LEVELS = {"dim": 0.3, "dark": 0.5, "very_dark": 0.65}
SPLIT = "test"
SEED = 42

def darken(img, level, rng):
    x = img.astype(np.float32) / 255.0
    atten = np.array([1.0 - 0.4*level, 1.0 - 0.7*level, 1.0 - 0.95*level], dtype=np.float32)
    x = x * atten
    x = np.power(x, 1.0 + 1.5*level) * (1.0 - 0.6*level)
    x = np.clip(x + rng.normal(0, 0.01 + 0.04*level, x.shape).astype(np.float32), 0, 1)
    return (x * 255).astype(np.uint8)

names_block = open(f"{SRC}/trashcan_full.yaml").read().split("names:")[1]

for name, level in LEVELS.items():
    out = f"dataset/trashcan_full_{name}"
    os.makedirs(f"{out}/images/{SPLIT}", exist_ok=True)
    os.makedirs(f"{out}/labels/{SPLIT}", exist_ok=True)
    rng = np.random.default_rng(SEED)
    for fname in tqdm(os.listdir(f"{SRC}/images/{SPLIT}"), desc=name):
        img = cv2.imread(f"{SRC}/images/{SPLIT}/{fname}")
        if img is None:
            continue
        cv2.imwrite(f"{out}/images/{SPLIT}/{fname}", darken(img, level, rng))
        stem = os.path.splitext(fname)[0]
        shutil.copy(f"{SRC}/labels/{SPLIT}/{stem}.txt", f"{out}/labels/{SPLIT}/{stem}.txt")
    with open(f"{out}/trashcan_full_{name}.yaml", "w") as f:
        f.write(f"path: {out}\ntrain: images/{SPLIT}\nval: images/{SPLIT}\ntest: images/{SPLIT}\nnames:{names_block}")
print("done")