"""History storage: SQLite database + saved files, with the /api/runs routes."""
import json, os, shutil, sqlite3, time, uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

STORAGE = Path(os.getenv("STORAGE_DIR", Path(__file__).resolve().parent / "storage"))
STORAGE.mkdir(parents=True, exist_ok=True)
DB = STORAGE / "history.db"
router = APIRouter(prefix="/api/runs")


def q(sql, args=(), one=False, write=False):
    c = sqlite3.connect(DB); c.row_factory = sqlite3.Row
    try:
        cur = c.execute(sql, args)
        if write:
            c.commit(); return None
        rows = cur.fetchall()
        return (rows[0] if rows else None) if one else rows
    finally:
        c.close()


q("""CREATE TABLE IF NOT EXISTS runs (id TEXT PRIMARY KEY, kind TEXT, name TEXT, created REAL,
     conf REAL, waste INT, animal INT, other INT, file TEXT, data TEXT)""", write=True)


def save_run(kind, name, conf, data, src):
    """Save one detection run. Image counts use a 0.25 threshold; video counts use tracked objects."""
    items = [d for d in data.get("detections", []) if d["confidence"] >= 0.25] if kind == "image" else data.get("objects", [])
    n = {g: sum(1 for d in items if d["group"] == g) for g in ("waste", "animal", "other")}
    rid = uuid.uuid4().hex[:10]
    fname = rid + Path(src).suffix.lower()
    shutil.copy(src, STORAGE / fname)
    q("INSERT INTO runs VALUES (?,?,?,?,?,?,?,?,?,?)",
      (rid, kind, name or "unnamed", time.time(), conf, n["waste"], n["animal"], n["other"], fname, json.dumps(data)), write=True)
    return rid


def _get(rid):
    r = q("SELECT * FROM runs WHERE id=?", (rid,), one=True)
    if not r: raise HTTPException(404, "Run not found")
    return r


@router.get("")
def list_runs(limit: int = 100):
    rows = q("SELECT id,kind,name,created,waste,animal,other FROM runs ORDER BY created DESC LIMIT ?", (limit,))
    return [dict(r) for r in rows]


@router.get("/{rid}")
def get_run(rid: str):
    r = dict(_get(rid)); r["data"] = json.loads(r["data"]); return r


@router.get("/{rid}/file")
def get_file(rid: str):
    return FileResponse(STORAGE / _get(rid)["file"])


@router.delete("/{rid}")
def delete_run(rid: str):
    r = _get(rid)
    (STORAGE / r["file"]).unlink(missing_ok=True)
    q("DELETE FROM runs WHERE id=?", (rid,), write=True)
    return {"deleted": rid}