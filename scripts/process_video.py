import cv2, csv
from collections import defaultdict, Counter
from ultralytics import YOLO


MODEL = "models/yolo26l_full_darkaware_best.pt"   # change to the new full model later
VIDEO = "test_video.mp4"
OUT_VIDEO = "result_video.mp4"
OUT_CSV = "trash_summary.csv"
CONF = 0.25
MIN_FRAMES = 3          # an object must appear in at least 3 frames to be counted

model = YOLO(MODEL)
names = model.names

cap = cv2.VideoCapture(VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS) or 15
w, h = int(cap.get(3)), int(cap.get(4))
cap.release()
writer = cv2.VideoWriter(OUT_VIDEO, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

seen = defaultdict(Counter)    # track id -> {class name: number of frames}

def color_for(name):
    if name.startswith("trash"):
        return (0, 0, 255)       # red (BGR)
    if name.startswith("animal") or name == "plant":
        return (0, 140, 255)     # orange
    return (255, 120, 0)         # blue for rov and others

for r in model.track(source=VIDEO, tracker="bytetrack.yaml", conf=CONF,
                     imgsz=640, stream=True, persist=True, verbose=False):
    frame = r.orig_img.copy()
    if r.boxes is not None and r.boxes.id is not None:
        for box, tid, cls in zip(r.boxes.xyxy.cpu().numpy(),
                                 r.boxes.id.int().cpu().tolist(),
                                 r.boxes.cls.int().cpu().tolist()):
            name = names[cls]
            seen[tid][name] += 1
            x1, y1, x2, y2 = map(int, box)
            col = color_for(name)
            cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
            cv2.putText(frame, f"{name} #{tid}", (x1, max(y1 - 6, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
    
    writer.write(frame)
writer.release()

# count unique objects (each track id counted once, using its most frequent class)
counts = Counter()
for tid, c in seen.items():
    if sum(c.values()) >= MIN_FRAMES:
        counts[c.most_common(1)[0][0]] += 1

with open(OUT_CSV, "w", newline="") as f:
    wr = csv.writer(f)
    wr.writerow(["class", "unique_objects", "type"])
    for n, k in sorted(counts.items(), key=lambda x: -x[1]):
        wr.writerow([n, k, "trash" if n.startswith("trash") else "other"])

print("\nUnique objects found:")
for n, k in sorted(counts.items(), key=lambda x: -x[1]):
    print(f"  {n}: {k}")
print("\nSaved", OUT_VIDEO, "and", OUT_CSV)