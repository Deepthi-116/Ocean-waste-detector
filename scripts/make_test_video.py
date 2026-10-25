import cv2, os, random
from logo import stamp

SRC = "dataset/trashcan_full/images/test"   # or dataset/trashcan_full_dark/images/test for a dark video
OUT = "test_video.mp4"
N_FRAMES = 150          # about 10 seconds at 15 fps
FPS = 15
SIZE = (640, 360)

random.seed(1)
files = sorted(os.listdir(SRC))
start = random.randint(0, len(files) - N_FRAMES)
files = files[start:start + N_FRAMES]       # consecutive file names are often neighboring frames

writer = cv2.VideoWriter(OUT, cv2.VideoWriter_fourcc(*"mp4v"), FPS, SIZE)
for f in files:
    img = cv2.resize(cv2.imread(os.path.join(SRC, f)), SIZE)
    writer.write(stamp(img))
writer.release()
print("Saved", OUT, "with", len(files), "frames")