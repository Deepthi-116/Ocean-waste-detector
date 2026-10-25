import cv2, os, random, yaml

ROOT = "dataset/trashcan_full"
IMG_DIR = f"{ROOT}/images/train"
LBL_DIR = f"{ROOT}/labels/train"
NAMES = yaml.safe_load(open(f"{ROOT}/trashcan_full.yaml"))["names"]

def classes_in(fname):
    p = os.path.join(LBL_DIR, os.path.splitext(fname)[0] + ".txt")
    return {int(l.split()[0]) for l in open(p) if l.strip()}

files = os.listdir(IMG_DIR)
has_animal = [f for f in files if any(NAMES[c].startswith("animal") for c in classes_in(f))]
has_both = [f for f in has_animal if any(NAMES[c].startswith("trash") for c in classes_in(f))]

pick = random.sample(has_both, min(3, len(has_both))) + random.sample(files, 3)

for i, fname in enumerate(pick):
    img = cv2.imread(os.path.join(IMG_DIR, fname))
    h, w = img.shape[:2]
    lbl = os.path.join(LBL_DIR, os.path.splitext(fname)[0] + ".txt")
    for line in open(lbl):
        c, xc, yc, bw, bh = line.split()
        c, xc, yc, bw, bh = int(c), float(xc), float(yc), float(bw), float(bh)
        x1, y1 = int((xc - bw/2) * w), int((yc - bh/2) * h)
        x2, y2 = int((xc + bw/2) * w), int((yc + bh/2) * h)
        color = (0, 255, 0) if NAMES[c].startswith("trash") else (0, 140, 255)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, NAMES[c], (x1 + 3, y1 + 18),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    cv2.imwrite(f"check_full_{i}.jpg", img)
print("Saved check_full_0.jpg ... check_full_5.jpg")