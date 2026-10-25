"""Does brightening dark images BEFORE detection help the YOLO26 dark-aware model?
Run from the project root:  python scripts/enhance_eval.py
Writes: enhancement_results.csv and enhance_preview.jpg"""
import cv2, re, shutil, numpy as np, pandas as pd
from pathlib import Path
from ultralytics import YOLO

MODEL = "models/yolo26l_full_darkaware_best.pt"
CONDS = {"dim": "dataset/trashcan_full_dim", "dark": "dataset/trashcan_full_dark",
         "very_dark": "dataset/trashcan_full_very_dark"}      # add "bright": "dataset/trashcan_full" as a control
WORK = Path("enh_eval")


def enhance(img):
    """1) undo the blue-green cast (gray-world, gain capped), 2) adaptive gamma brighten, 3) CLAHE contrast."""
    f = img.astype(np.float32)
    m = f.reshape(-1, 3).mean(0) + 1e-6
    f = np.clip(f * np.clip(m.mean() / m, 1, 6), 0, 255)
    mean = max(f.mean() / 255, 1e-3)
    gamma = float(np.clip(np.log(0.4) / np.log(mean), 0.3, 1.0))   # aim for a mean brightness of about 0.4
    img = (255 * (f / 255) ** gamma).astype(np.uint8)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def make_enhanced(src, dst, yaml_name):
    (dst / "images/test").mkdir(parents=True, exist_ok=True)
    for p in (src / "images/test").iterdir():
        cv2.imwrite(str(dst / "images/test" / p.name), enhance(cv2.imread(str(p))))
    if not (dst / "labels").exists():
        shutil.copytree(src / "labels", dst / "labels")
    text = (src / yaml_name).read_text()
    (dst / yaml_name).write_text(re.sub(r"(?m)^path:.*$", f"path: {dst.as_posix()}", text, count=1))


model = YOLO(MODEL)
rows, tiles = [], []
for cond, folder in CONDS.items():
    src, yaml_name = Path(folder), f"{Path(folder).name}.yaml"
    dst = WORK / Path(folder).name
    make_enhanced(src, dst, yaml_name)
    for label, root in (("original", src), ("enhanced", dst)):
        r = model.val(data=str(root / yaml_name), split="test", imgsz=640, batch=8, plots=False, verbose=False)
        trash = [i for i, n in r.names.items() if n.startswith("trash")]
        rows.append(dict(condition=cond, input=label, mAP50=r.box.map50, mAP50_95=r.box.map,
                         trash_only_mAP50_95=float(np.mean([r.box.maps[i] for i in trash]))))
        print(rows[-1])
    name = sorted(p.name for p in (src / "images/test").iterdir())[5]
    tiles.append(np.vstack([cv2.resize(cv2.imread(str(src / "images/test" / name)), (320, 180)),
                            cv2.resize(cv2.imread(str(dst / "images/test" / name)), (320, 180))]))

df = pd.DataFrame(rows).round(3)
df.to_csv("enhancement_results.csv", index=False)
cv2.imwrite("enhance_preview.jpg", np.hstack(tiles))
print(df.to_string(index=False))
print("Saved enhancement_results.csv and enhance_preview.jpg (top row original, bottom row enhanced)")