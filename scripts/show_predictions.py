import cv2, os, random, yaml
import numpy as np
from ultralytics import YOLO

# ---------- SETTINGS ----------
WEIGHTS = "models/yolo26l_darkaware_best.pt"   # your best.pt
SET = "dataset/trashcan"          # try dataset/trashcan_dark or trashcan_very_dark too
N_IMAGES = 6
CONF = 0.25
# ------------------------------

model = YOLO(WEIGHTS)
names = yaml.safe_load(open("dataset/trashcan/trashcan.yaml"))["names"]

img_dir = f"{SET}/images/test"
lbl_dir = f"{SET}/labels/test"
random.seed(1)
files = random.sample(os.listdir(img_dir), N_IMAGES)

def draw_gt(img, lbl_path):
    h, w = img.shape[:2]
    for line in open(lbl_path):
        c, xc, yc, bw, bh = line.split()
        c, xc, yc, bw, bh = int(c), float(xc), float(yc), float(bw), float(bh)
        x1, y1 = int((xc - bw / 2) * w), int((yc - bh / 2) * h)
        x2, y2 = int((xc + bw / 2) * w), int((yc + bh / 2) * h)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, names[c], (x1, max(y1 - 5, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return img

os.makedirs("mentor_images", exist_ok=True)
for i, fname in enumerate(files):
    path = f"{img_dir}/{fname}"
    gt = draw_gt(cv2.imread(path), f"{lbl_dir}/{os.path.splitext(fname)[0]}.txt")
    pred = model.predict(path, conf=CONF, verbose=False)[0].plot()   # red/blue boxes

    cv2.putText(gt, "Ground truth", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(pred, "Model prediction", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.imwrite(f"mentor_images/compare_{i}.jpg", np.hstack([gt, pred]))

print("Saved to mentor_images/")