import json, os, random, shutil
from collections import defaultdict, Counter
from tqdm import tqdm

RAW = "dataset/instance_version"
JSON_FILES = {"train": f"{RAW}/instances_train_trashcan.json",
              "val":   f"{RAW}/instances_val_trashcan.json"}
IMAGE_DIRS = {"train": f"{RAW}/train", "val": f"{RAW}/val"}
OUT = "dataset/trashcan_full"
SPLIT = (0.70, 0.15, 0.15)
SEED = 42

random.seed(SEED)
data = {p: json.load(open(j)) for p, j in JSON_FILES.items()}

# collect every class name and count boxes
id_to_name, counts = {}, Counter()
for d in data.values():
    for c in d["categories"]:
        id_to_name[c["id"]] = c["name"]
for d in data.values():
    for a in d["annotations"]:
        counts[id_to_name[a["category_id"]]] += 1

KEEP = sorted(set(id_to_name.values()))      # ALL classes
name_to_new = {n: i for i, n in enumerate(KEEP)}
print("Classes:")
for n in KEEP:
    print(f"  {name_to_new[n]:2d} {n}: {counts[n]}")

items = []
for part, d in data.items():
    cat_name = {c["id"]: c["name"] for c in d["categories"]}
    imgs = {im["id"]: im for im in d["images"]}
    boxes = defaultdict(list)
    for a in d["annotations"]:
        im = imgs[a["image_id"]]
        W, H = im["width"], im["height"]
        x, y, w, h = a["bbox"]
        boxes[a["image_id"]].append((name_to_new[cat_name[a["category_id"]]],
            (x + w/2)/W, (y + h/2)/H, w/W, h/H))
    for iid, im in imgs.items():
        if iid in boxes:
            path = os.path.join(IMAGE_DIRS[part], os.path.basename(im["file_name"]))
            if os.path.exists(path):
                items.append((path, boxes[iid]))

print("Images kept:", len(items))
random.shuffle(items)
n = len(items); n_tr, n_va = int(n*SPLIT[0]), int(n*SPLIT[1])
splits = {"train": items[:n_tr], "val": items[n_tr:n_tr+n_va], "test": items[n_tr+n_va:]}

for split, rows in splits.items():
    os.makedirs(f"{OUT}/images/{split}", exist_ok=True)
    os.makedirs(f"{OUT}/labels/{split}", exist_ok=True)
    for img_path, bxs in tqdm(rows, desc=split):
        fname = os.path.basename(img_path)
        shutil.copy(img_path, f"{OUT}/images/{split}/{fname}")
        with open(f"{OUT}/labels/{split}/{os.path.splitext(fname)[0]}.txt", "w") as f:
            for c, xc, yc, w, h in bxs:
                f.write(f"{c} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

with open(f"{OUT}/trashcan_full.yaml", "w") as f:
    f.write(f"path: {OUT}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n")
    for i, n_ in enumerate(KEEP):
        f.write(f"  {i}: {n_}\n")
print({k: len(v) for k, v in splits.items()})