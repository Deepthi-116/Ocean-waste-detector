import os, torch

ROOT = "ocean_weights"
for folder in sorted(os.listdir(ROOT)):
    p = os.path.join(ROOT, folder, "best.pt")
    if not os.path.exists(p):
        continue
    ck = torch.load(p, map_location="cpu", weights_only=False)
    a = ck.get("train_args", {})
    print(f"{folder:20s} run name: {str(a.get('name')):22s} data: {a.get('data')}")