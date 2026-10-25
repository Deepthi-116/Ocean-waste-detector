import json
from collections import Counter

# Check these file names match exactly what you have
JSON_FILES = [
    "dataset/instance_version/instances_train_trashcan.json",
    "dataset/instance_version/instances_val_trashcan.json",
]

total_images = 0
counts = Counter()
id_to_name = {}

for path in JSON_FILES:
    with open(path) as f:
        data = json.load(f)
    total_images += len(data["images"])
    for c in data["categories"]:
        id_to_name[c["id"]] = c["name"]
    for a in data["annotations"]:
        counts[a["category_id"]] += 1

print("Total images:", total_images)
print("\nClass id -> name -> number of boxes")
for cid, name in sorted(id_to_name.items()):
    print(cid, name, counts[cid])