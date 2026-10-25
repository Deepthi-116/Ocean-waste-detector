import { useCallback, useEffect, useRef, useState } from "react";
import History from "./History.jsx";

const COLORS = { waste: "#ff6a3d", animal: "#5ad1e6", other: "#c9b28a" };
const LABELS = { waste: "Waste", animal: "Animals", other: "Plants & ROV" };

async function post(url, file, fields) {
  const f = new FormData(); f.append("file", file);
  Object.entries(fields).forEach(([k, v]) => f.append(k, v));
  const r = await fetch(url, { method: "POST", body: f });
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
  return r.json();
}
const toCsv = (rows) => { const k = Object.keys(rows[0] || {}); return [k.join(","), ...rows.map((r) => k.map((x) => JSON.stringify(r[x] ?? "")).join(","))].join("\n"); };
const saveCsv = (name, rows) => { const a = document.createElement("a"); a.href = URL.createObjectURL(new Blob([toCsv(rows)], { type: "text/csv" })); a.download = name; a.click(); };
const count = (items) => items.reduce((o, d) => ({ ...o, [d.group]: (o[d.group] || 0) + 1 }), {});

function Counts({ items }) {
  const c = count(items);
  return <div className="counts">{Object.keys(LABELS).map((g) => <div key={g} style={{ borderColor: COLORS[g] }}><strong>{c[g] || 0}</strong><span>{LABELS[g]}</span></div>)}</div>;
}
function Dot({ g }) { return <span className="dot" style={{ background: COLORS[g] }} />; }

function Overlay({ src, dets, w, h, bright, hover }) {
  const img = useRef(), cv = useRef();
  const draw = useCallback(() => {
    const im = img.current, c = cv.current; if (!im || !c) return;
    c.width = im.clientWidth; c.height = im.clientHeight;
    const s = c.width / w, g = c.getContext("2d"); g.font = "12px sans-serif";
    dets.forEach((d, i) => {
      const [x1, y1, x2, y2] = d.box.map((v) => v * s), col = COLORS[d.group];
      g.strokeStyle = col; g.lineWidth = i === hover ? 4 : 2; g.strokeRect(x1, y1, x2 - x1, y2 - y1);
      const t = `${d.class_name} ${d.confidence.toFixed(2)}`, ty = Math.max(y1 - 16, 0);
      g.fillStyle = col; g.fillRect(x1, ty, g.measureText(t).width + 6, 16); g.fillStyle = "#061218"; g.fillText(t, x1 + 3, ty + 12);
    });
  }, [dets, w, hover]);
  useEffect(() => { draw(); const ro = new ResizeObserver(draw); ro.observe(img.current); return () => ro.disconnect(); }, [draw]);
  return <div className="stage"><img ref={img} src={src} onLoad={draw} alt="Uploaded frame" style={{ filter: `brightness(${bright}) contrast(1.1)` }} /><canvas ref={cv} /></div>;
}

function ImageMode({ file, model }) {
  const [res, setRes] = useState(null), [err, setErr] = useState(""), [busy, setBusy] = useState(true);
  const [conf, setConf] = useState(0.25), [bright, setBright] = useState(1), [on, setOn] = useState({ waste: true, animal: true, other: true }), [hover, setHover] = useState(-1);
  const [url, setUrl] = useState("");
  useEffect(() => {
    const u = URL.createObjectURL(file); setUrl(u); setBusy(true); setErr(""); setRes(null);
    post("/api/detect/image", file, { conf: 0.1, model }).then(setRes).catch((e) => setErr(e.message)).finally(() => setBusy(false));
    return () => URL.revokeObjectURL(u);
  }, [file, model]);
  const vis = res ? res.detections.filter((d) => d.confidence >= conf && on[d.group]) : [];
  return (<div className="grid">
    <div className="panel">{url && <Overlay src={url} dets={vis} w={res?.width || 1} h={res?.height || 1} bright={bright} hover={hover} />}
      {busy && <p className="note">Detecting…</p>}{err && <p className="err">{err}</p>}</div>
    <div className="panel">
      <Counts items={vis} />
      <label>Confidence threshold <b>{conf.toFixed(2)}</b></label><input type="range" min="0.1" max="0.9" step="0.05" value={conf} onChange={(e) => setConf(+e.target.value)} />
      <label>Preview brightness <b>{bright.toFixed(1)}×</b></label><input type="range" min="1" max="4" step="0.1" value={bright} onChange={(e) => setBright(+e.target.value)} />
      <div className="chk">{Object.keys(LABELS).map((g) => <label key={g}><input type="checkbox" checked={on[g]} onChange={() => setOn({ ...on, [g]: !on[g] })} /> <Dot g={g} />{LABELS[g]}</label>)}</div>
      {res && <p className="note">Inference {res.inference_ms} ms. Brightness only changes the preview, not the detection.</p>}
      <div className="scroll"><table><thead><tr><th>Class</th><th>Conf.</th></tr></thead><tbody>
        {vis.map((d, i) => <tr key={i} className={i === hover ? "h" : ""} onMouseEnter={() => setHover(i)} onMouseLeave={() => setHover(-1)}><td><Dot g={d.group} />{d.class_name}</td><td>{d.confidence.toFixed(2)}</td></tr>)}
      </tbody></table>{res && !vis.length && <p className="note">Nothing above this threshold. Lower the slider.</p>}</div>
      {vis.length > 0 && <button className="btn" onClick={() => saveCsv("detections.csv", vis.map((d) => ({ class: d.class_name, group: d.group, confidence: d.confidence, x1: d.box[0], y1: d.box[1], x2: d.box[2], y2: d.box[3] })))}>Download CSV</button>}
    </div></div>);
}

function VideoMode({ file, model }) {
  const [conf, setConf] = useState(0.25), [stride, setStride] = useState(1), [bright, setBright] = useState(1);
  const [job, setJob] = useState(null), [err, setErr] = useState(""), [state, setState] = useState({ status: "idle", progress: 0 });
  useEffect(() => { setJob(null); setState({ status: "idle", progress: 0 }); setErr(""); }, [file]);
  useEffect(() => {
    if (!job) return;
    const t = setInterval(async () => {
      try { const r = await fetch(`/api/jobs/${job}`); const j = await r.json(); if (!r.ok) throw new Error(j.detail); setState(j); if (j.status === "done" || j.status === "error") clearInterval(t); }
      catch (e) { setErr(e.message); clearInterval(t); }
    }, 1000);
    return () => clearInterval(t);
  }, [job]);
  const start = async () => { setErr(""); setState({ status: "queued", progress: 0 }); try { setJob((await post("/api/detect/video", file, { conf, stride, model })).job_id); } catch (e) { setErr(e.message); setState({ status: "idle", progress: 0 }); } };
  const running = ["queued", "running"].includes(state.status), r = state.result;
  return (<div className="grid">
    <div className="panel">
      {state.status === "done" ? <div className="stage"><video src={`/api/jobs/${job}/video`} controls style={{ filter: `brightness(${bright}) contrast(1.1)` }} /></div> : <p className="note">{file.name}</p>}
      {state.status === "done" && !r.browser_playable && <p className="err">This browser cannot play the result. Use Download video.</p>}
      {running && <><div className="bar"><i style={{ width: `${Math.round((state.progress || 0) * 100)}%` }} /></div><p className="note">{state.status === "queued" ? "Waiting for the detector…" : `Processing ${Math.round((state.progress || 0) * 100)}%. CPU runs can take several minutes.`}</p></>}
      {(err || state.status === "error") && <p className="err">{err || state.error}</p>}
    </div>
    <div className="panel">
      {state.status !== "done" && <>
        <label>Confidence threshold <b>{conf.toFixed(2)}</b></label><input type="range" min="0.1" max="0.9" step="0.05" value={conf} disabled={running} onChange={(e) => setConf(+e.target.value)} />
        <label>Process every Nth frame <b>{stride}</b></label><input type="range" min="1" max="5" step="1" value={stride} disabled={running} onChange={(e) => setStride(+e.target.value)} />
        <p className="note">Higher N is faster but may miss brief objects.</p>
        <button className="btn" disabled={running} onClick={start}>Detect in video</button></>}
      {r && <>
        <Counts items={r.objects} />
        <p className="note">Unique objects, each counted once by tracking. {r.duration_s}s video, processed in {r.seconds_taken}s.</p>
        <label>Preview brightness <b>{bright.toFixed(1)}×</b></label><input type="range" min="1" max="4" step="0.1" value={bright} onChange={(e) => setBright(+e.target.value)} />
        <div className="scroll"><table><thead><tr><th>Class</th><th>First seen</th><th>Conf.</th></tr></thead><tbody>
          {r.objects.map((o) => <tr key={o.track_id}><td><Dot g={o.group} />{o.class_name}</td><td>{o.first_seen_s}s</td><td>{o.max_confidence}</td></tr>)}</tbody></table>
          {!r.objects.length && <p className="note">No stable objects found. Try a lower threshold.</p>}</div>
        <a className="btn" href={`/api/jobs/${job}/video`} download="detections.mp4">Download video</a>
        {r.objects.length > 0 && <button className="btn alt" onClick={() => saveCsv("tracked_objects.csv", r.objects)}>Download CSV</button>}
      </>}
    </div></div>);
}

export default function App() {
  const [mode, setMode] = useState("image"), [file, setFile] = useState(null), [health, setHealth] = useState(null), [over, setOver] = useState(false), [model, setModel] = useState("yolo26");
  useEffect(() => { fetch("/api/health").then((r) => r.json()).then((h) => { setHealth(h); if (h.default) setModel(h.default); }).catch(() => setHealth({ model_loaded: false, error: "Cannot reach the backend. Start it with: uvicorn main:app --port 8000" })); }, []);
  const pick = (f) => { if (!f) return; setFile(f); };
  const accept = mode === "image" ? "image/*" : "video/*";
  return (<div className="wrap">
    <header><h1>Deep Ocean Waste Detection Under Low-Light Conditions</h1>
      <div>
        <span className={"pill " + (health?.model_loaded ? "ok" : "")}>{health ? (health.model_loaded ? `Model ready (${health.device})` : "Model offline") : "Checking model…"}</span>
        {health?.models && <select aria-label="Model" value={model} onChange={(e) => setModel(e.target.value)} style={{ marginLeft: 8, background: "#0d2530", color: "#dcecee", border: "1px solid #1d4251", borderRadius: 6, padding: "4px 8px" }}>
          {health.models.map((m) => <option key={m.key} value={m.key} disabled={!m.available}>{m.label}{m.available ? "" : " (weights missing)"}</option>)}
        </select>}
      </div>
    </header>
    {health && !health.model_loaded && <p className="err">{health.error}</p>}
    <div className="tabs" role="tablist">{["image", "video", "history"].map((m) => <button key={m} role="tab" aria-selected={mode === m} className={mode === m ? "on" : ""} onClick={() => { setMode(m); setFile(null); }}>{m === "image" ? "Image" : m === "video" ? "Video" : "History"}</button>)}</div>
    <label className={"drop " + (over ? "over" : "")} style={{ margin: 0, display: mode === "history" ? "none" : "block" }} onDragOver={(e) => { e.preventDefault(); setOver(true); }} onDragLeave={() => setOver(false)} onDrop={(e) => { e.preventDefault(); setOver(false); pick(e.dataTransfer.files[0]); }}>
      <b>{file ? file.name : `Drop ${mode === "image" ? "an image" : "a video"} here`}</b>{file ? "Click to choose another file" : "or click to browse"}
      <input type="file" hidden accept={accept} onChange={(e) => pick(e.target.files[0])} /></label>
    <div style={{ marginTop: 16 }}>{file && (mode === "image" ? <ImageMode file={file} model={model} /> : <VideoMode file={file} model={model} />)}</div>
    {mode === "history" && <History ui={{ Overlay, Counts, Dot, saveCsv }} />}
  </div>);
}