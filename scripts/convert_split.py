import json, os, random, shutil
from collections import defaultdict, Counter
from tqdm import tqdm

# ---------- SETTINGS ----------
RAW = "dataset/instance_version"
JSON_FILES = {
    "train": f"{RAW}/instances_train_trashcan.json",
    "val":   f"{RAW}/instances_val_trashcan.json",
}
IMAGE_DIRS = {"train": f"{RAW}/train", "val": f"{RAW}/val"}

OUT = "dataset/all_trashcan"

MERGE_ALL = False       # True = one class "trash"; False = keep each trash_* class
MIN_BOXES = 50          # trash classes with fewer boxes than this are dropped (multi-class mode)
SPLIT = (0.70, 0.15, 0.15)
SEED = 42
# ------------------------------

random.seed(SEED)

# 1) Load both JSON files
data = {}
for part, jpath in JSON_FILES.items():
    with open(jpath) as f:
        data[part] = json.load(f)

# 2) Find trash classes and count their boxes
id_to_name = {}
counts = Counter()
for part, d in data.items():
    for c in d["categories"]:
        id_to_name[c["id"]] = c["name"]
    for a in d["annotations"]:
        counts[id_to_name[a["category_id"]]] += 1

trash_names = sorted(n for n in set(id_to_name.values()) if n.startswith("trash"))
print("Trash classes found:")
for n in trash_names:
    print(f"  {n}: {counts[n]}")

if MERGE_ALL:
    KEEP = ["trash"]
    name_to_new = {n: 0 for n in trash_names}
else:
    KEEP = [n for n in trash_names if counts[n] >= MIN_BOXES]
    dropped = [n for n in trash_names if counts[n] < MIN_BOXES]
    name_to_new = {n: i for i, n in enumerate(KEEP)}
    print("Dropped (too rare):", dropped)

print("Final classes:", KEEP)

# 3) Build the list of (image_path, boxes)
items = []
for part, d in data.items():
    cat_name = {c["id"]: c["name"] for c in d["categories"]}
    imgs = {im["id"]: im for im in d["images"]}
    boxes = defaultdict(list)
    for a in d["annotations"]:
        name = cat_name[a["category_id"]]
        if name not in name_to_new:
            continue
        im = imgs[a["image_id"]]
        W, H = im["width"], im["height"]
        x, y, w, h = a["bbox"]
        boxes[a["image_id"]].append(
            (name_to_new[name], (x + w / 2) / W, (y + h / 2) / H, w / W, h / H))
    for iid, im in imgs.items():
        if iid in boxes:
            path = os.path.join(IMAGE_DIRS[part], os.path.basename(im["file_name"]))
            if os.path.exists(path):
                items.append((path, boxes[iid]))

print("Images kept:", len(items))

# 4) Shuffle and split
random.shuffle(items)
n = len(items)
n_train, n_val = int(n * SPLIT[0]), int(n * SPLIT[1])
splits = {
    "train": items[:n_train],
    "val":   items[n_train:n_train + n_val],
    "test":  items[n_train + n_val:],
}

# 5) Write images and labels
for split, rows in splits.items():
    os.makedirs(f"{OUT}/images/{split}", exist_ok=True)
    os.makedirs(f"{OUT}/labels/{split}", exist_ok=True)
    for img_path, bxs in tqdm(rows, desc=split):
        fname = os.path.basename(img_path)
        shutil.copy(img_path, f"{OUT}/images/{split}/{fname}")
        stem = os.path.splitext(fname)[0]
        with open(f"{OUT}/labels/{split}/{stem}.txt", "w") as f:
            for c, xc, yc, w, h in bxs:
                f.write(f"{c} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

# 6) Write the YAML file automatically
with open(f"{OUT}/trashcan.yaml", "w") as f:
    f.write(f"path: {OUT}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n")
    for i, n in enumerate(KEEP):
        f.write(f"  {i}: {n}\n")

print({k: len(v) for k, v in splits.items()})
print("Wrote", f"{OUT}/trashcan.yaml")