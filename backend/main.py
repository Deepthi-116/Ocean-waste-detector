"""Ocean waste detector API: image + video detection with tracking, history, and a model switcher."""
import os, time, uuid, shutil, subprocess, tempfile, threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from db import router as history_router, save_run
try:
    from logo import stamp
except ImportError:                       # logo.py is optional
    stamp = lambda frame: frame

MODELS_DIR = Path(os.getenv("MODELS_DIR", Path(__file__).resolve().parent.parent / "models"))
MODEL_FILES = {                           # key: (label shown in the app, weights file inside the models folder)
    "yolo26": ("YOLO26-l", "yolo26l_full_darkaware_best.pt"),
    "yolo11": ("YOLO11-l", "yolo11l_full_darkaware_best.pt"),
    "rtdetr": ("RT-DETR-l", "rtdetr_l_full_darkaware_best.pt"),
}
DEFAULT_MODEL = "yolo26"
MAX_MB = int(os.getenv("MAX_UPLOAD_MB", "300"))
MIN_TRACK_FRAMES = int(os.getenv("MIN_TRACK_FRAMES", "3"))   # ignore flickering tracks
IMGSZ = int(os.getenv("IMGSZ", "640"))
STAMP_LOGO = os.getenv("LOGO_STAMP", "0") == "1"            # set to 1 only if the input video has no logo
JOB_TTL = 3600
IMG_EXT, VID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}, {".mp4", ".mov", ".avi", ".mkv", ".webm"}
WORK = Path(tempfile.gettempdir()) / "ocean_waste_jobs"; WORK.mkdir(exist_ok=True)
COLORS = {"waste": (61, 106, 255), "animal": (230, 209, 90), "other": (138, 178, 201)}  # BGR


def group_of(name):
    return "waste" if name.startswith("trash") else "animal" if name.startswith("animal") else "other"


def device():
    try:
        import torch
        return 0 if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


class Detector:
    def __init__(self, path, dev, rtdetr):
        from ultralytics import YOLO, RTDETR
        cls = RTDETR if rtdetr else YOLO
        self.device, self.lock = dev, threading.Lock()
        # two instances: tracking attaches state to a model, which would alter plain predictions
        self.img, self.vid = cls(str(path)), cls(str(path))

    def _dets(self, r):
        if r.boxes is None or len(r.boxes) == 0: return []
        ids = r.boxes.id.int().tolist() if r.boxes.id is not None else [None] * len(r.boxes)
        out = []
        for b, c, k, t in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.int().tolist(), ids):
            n = r.names[k]
            out.append(dict(class_id=k, class_name=n, group=group_of(n), confidence=round(c, 4),
                            box=[round(v, 1) for v in b], track_id=t))
        return out

    def predict(self, img, conf):
        with self.lock:
            return self._dets(self.img.predict(img, conf=conf, imgsz=IMGSZ, device=self.device, verbose=False)[0])

    def track(self, frame, conf, first):
        r = self.vid.track(frame, persist=not first, conf=conf, imgsz=IMGSZ, device=self.device,
                           tracker="bytetrack.yaml", verbose=False)[0]
        return self._dets(r)


DETS, DET_LOCK = {}, threading.Lock()
JOBS, pool = {}, ThreadPoolExecutor(1)  # one video at a time


def available():
    return [k for k, (_, f) in MODEL_FILES.items() if (MODELS_DIR / f).exists()]


def get_det(key):
    """Load a model the first time it is used, then reuse it."""
    if key not in MODEL_FILES: raise HTTPException(400, f"Unknown model '{key}'")
    path = MODELS_DIR / MODEL_FILES[key][1]
    if not path.exists(): raise HTTPException(503, f"Weights not found: {path.name}. Copy it into the models folder.")
    with DET_LOCK:
        if key not in DETS:
            try: DETS[key] = Detector(path, device(), key == "rtdetr")
            except Exception as e: raise HTTPException(503, f"Could not load {MODEL_FILES[key][0]}: {e}")
    return DETS[key]


@asynccontextmanager
async def lifespan(app):
    try: get_det(DEFAULT_MODEL)          # warm up the default model
    except HTTPException: pass
    yield


app = FastAPI(title="Ocean Waste Detector", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
                   allow_methods=["*"], allow_headers=["*"])
app.include_router(history_router)


def save_upload(f, kinds, folder):
    ext = Path(f.filename or "").suffix.lower()
    if ext not in kinds:
        raise HTTPException(400, f"Unsupported file type '{ext}'. Use: {', '.join(sorted(kinds))}")
    path, size = folder / f"input{ext}", 0
    with open(path, "wb") as o:
        while chunk := f.file.read(1 << 20):
            size += len(chunk)
            if size > MAX_MB << 20: raise HTTPException(413, f"File is larger than {MAX_MB} MB")
            o.write(chunk)
    return path


def draw(frame, dets):
    for d in dets:
        x1, y1, x2, y2 = map(int, d["box"]); col = COLORS[d["group"]]
        cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
        label = f'{d["class_name"]} {d["confidence"]:.2f}' + (f' #{d["track_id"]}' if d["track_id"] is not None else "")
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        y0 = max(y1 - h - 6, 0)
        cv2.rectangle(frame, (x1, y0), (x1 + w + 6, y0 + h + 6), col, -1)
        cv2.putText(frame, label, (x1 + 3, y0 + h + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (26, 18, 6), 1, cv2.LINE_AA)


def reencode(src, dst):
    """mp4v is not playable in browsers; convert to H.264."""
    try:
        import imageio_ffmpeg
        r = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-i", str(src), "-c:v", "libx264", "-preset", "veryfast",
                            "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(dst)], capture_output=True)
        return r.returncode == 0 and dst.exists()
    except Exception:
        return False


def purge():
    for jid, j in list(JOBS.items()):
        if time.time() - j["created"] > JOB_TTL and j["status"] != "running":
            shutil.rmtree(j["dir"], ignore_errors=True); JOBS.pop(jid, None)


def run_video(jid, path, conf, stride, d):
    job, t0 = JOBS[jid], time.time(); job["status"] = "running"
    try:
        cap = cv2.VideoCapture(str(path)); fps = cap.get(cv2.CAP_PROP_FPS) or 25
        total, W, H = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), int(cap.get(3)), int(cap.get(4))
        raw = job["dir"] / "raw.mp4"
        wr = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
        tracks, last, i, first = {}, [], 0, True
        while True:
            ok, frame = cap.read()
            if not ok: break
            if i % stride == 0:
                last = d.track(frame, conf, first); first = False
                for x in last:
                    if x["track_id"] is None: continue
                    t = tracks.setdefault(x["track_id"], dict(votes=Counter(), first=i, last=i, n=0, mx=0.0))
                    t["votes"][x["class_name"]] += x["confidence"]; t["last"] = i; t["n"] += 1; t["mx"] = max(t["mx"], x["confidence"])
            draw(frame, last)
            if STAMP_LOGO: stamp(frame)
            wr.write(frame); i += 1
            if total > 0 and i % 5 == 0: job["progress"] = min(i / total, 0.99)
        cap.release(); wr.release()
        if i == 0: raise ValueError("No frames could be read from this video")
        out = job["dir"] / "result.mp4"; playable = reencode(raw, out)
        objs = []
        for tid, t in tracks.items():
            if t["n"] < MIN_TRACK_FRAMES: continue
            cn = t["votes"].most_common(1)[0][0]
            objs.append(dict(track_id=tid, class_name=cn, group=group_of(cn), first_seen_s=round(t["first"] / fps, 2),
                             last_seen_s=round(t["last"] / fps, 2), max_confidence=round(t["mx"], 3)))
        objs.sort(key=lambda o: o["first_seen_s"])
        result = dict(objects=objs, frames=i, fps=round(fps, 2), duration_s=round(i / fps, 1),
                      seconds_taken=round(time.time() - t0, 1), browser_playable=playable, model=job["model"])
        final = out if playable else raw
        result["run_id"] = save_run("video", job["name"], conf, result, final)
        job.update(status="done", progress=1.0, video=str(final), result=result)
    except Exception as e:
        job.update(status="error", error=str(e))


@app.get("/api/health")
def health():
    ok = available()
    return dict(model_loaded=bool(ok), device=str(device()),
                default=DEFAULT_MODEL if DEFAULT_MODEL in ok else (ok[0] if ok else None),
                models=[dict(key=k, label=v[0], available=k in ok) for k, v in MODEL_FILES.items()],
                error=None if ok else f"No weights found in {MODELS_DIR}. Copy your best.pt files there.")


@app.post("/api/detect/image")
def detect_image(file: UploadFile = File(...), conf: float = Form(0.15), model: str = Form(DEFAULT_MODEL)):
    d = get_det(model); folder = Path(tempfile.mkdtemp(dir=WORK))
    try:
        path = save_upload(file, IMG_EXT, folder)
        img = cv2.imread(str(path))
        if img is None: raise HTTPException(400, "Could not read this image")
        t = time.time(); dets = d.predict(img, conf)
        out = dict(width=img.shape[1], height=img.shape[0], inference_ms=round((time.time() - t) * 1000, 1),
                   detections=dets, model=model)
        out["run_id"] = save_run("image", f"{file.filename} [{MODEL_FILES[model][0]}]", conf, out, path)
        return out
    finally:
        shutil.rmtree(folder, ignore_errors=True)


@app.post("/api/detect/video")
def detect_video(file: UploadFile = File(...), conf: float = Form(0.25), stride: int = Form(1), model: str = Form(DEFAULT_MODEL)):
    d = get_det(model); purge()
    jid = uuid.uuid4().hex[:12]; folder = WORK / jid; folder.mkdir()
    path = save_upload(file, VID_EXT, folder)
    JOBS[jid] = dict(status="queued", progress=0.0, dir=folder, created=time.time(),
                     name=f"{file.filename} [{MODEL_FILES[model][0]}]", model=model)
    pool.submit(run_video, jid, path, conf, max(1, min(stride, 5)), d)
    return {"job_id": jid}


@app.get("/api/jobs/{jid}")
def job_status(jid: str):
    j = JOBS.get(jid)
    if not j: raise HTTPException(404, "Job not found. It may have expired.")
    return {k: j.get(k) for k in ("status", "progress", "error", "result")}


@app.get("/api/jobs/{jid}/video")
def job_video(jid: str):
    j = JOBS.get(jid)
    if not j or j["status"] != "done": raise HTTPException(404, "Video not ready")
    return FileResponse(j["video"], media_type="video/mp4")

STATIC = Path(__file__).resolve().parent / "static"
if STATIC.exists():
    app.mount("/", StaticFiles(directory=STATIC, html=True), name="ui")
    