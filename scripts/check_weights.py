import hashlib, os

ROOT = "ocean_weights"
for folder in sorted(os.listdir(ROOT)):
    p = os.path.join(ROOT, folder, "best.pt")
    if os.path.exists(p):
        h = hashlib.md5(open(p, "rb").read()).hexdigest()[:10]
        print(f"{folder:22s} {h}   {os.path.getsize(p)/1e6:.1f} MB")
    else:
        print(f"{folder:22s} best.pt NOT FOUND")