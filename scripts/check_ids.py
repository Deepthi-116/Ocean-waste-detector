import os, yaml
from collections import Counter

ROOT = "dataset/all_trashcan"
with open(f"{ROOT}/trashcan.yaml") as f:
    print("YAML names:", yaml.safe_load(f)["names"])

for split in ["train", "val", "test"]:
    d = f"{ROOT}/labels/{split}"
    if not os.path.isdir(d):
        print(split, "-> folder missing")
        continue
    ids = Counter()
    for fn in os.listdir(d):
        if fn.endswith(".txt"):
            for line in open(os.path.join(d, fn)):
                if line.strip():
                    ids[int(line.split()[0])] += 1
    print(split, "-> files:", len(os.listdir(d)), "| class ids:", dict(sorted(ids.items())))