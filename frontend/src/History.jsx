import { useEffect, useState } from "react";

const fmt = (t) => new Date(t * 1000).toLocaleString();

export default function History({ ui }) {
  const { Overlay, Counts, Dot, saveCsv } = ui;
  const [runs, setRuns] = useState(null), [open, setOpen] = useState(null), [err, setErr] = useState("");
  const [conf, setConf] = useState(0.25), [bright, setBright] = useState(1);

  const load = () => fetch("/api/runs").then((r) => r.json()).then(setRuns).catch((e) => setErr(e.message));
  useEffect(() => { load(); }, []);

  const view = async (id) => {
    try {
      const r = await fetch(`/api/runs/${id}`), j = await r.json();
      if (!r.ok) throw new Error(j.detail);
      setOpen(j); setConf(0.25); setBright(1);
    } catch (e) { setErr(e.message); }
  };
  const del = async (id) => {
    if (!window.confirm("Delete this run and its saved file?")) return;
    await fetch(`/api/runs/${id}`, { method: "DELETE" });
    if (open?.id === id) setOpen(null);
    load();
  };

  if (open) {
    const d = open.data, isImg = open.kind === "image", src = `/api/runs/${open.id}/file`;
    const items = isImg ? d.detections.filter((x) => x.confidence >= conf) : d.objects;
    return (<div>
      <button className="btn alt" style={{ marginTop: 0 }} onClick={() => setOpen(null)}>← Back to history</button>
      <p className="note">{open.name} · {fmt(open.created)}</p>
      <div className="grid">
        <div className="panel">{isImg
          ? <Overlay src={src} dets={items} w={d.width} h={d.height} bright={bright} hover={-1} />
          : <div className="stage"><video src={src} controls style={{ filter: `brightness(${bright}) contrast(1.1)` }} /></div>}</div>
        <div className="panel">
          <Counts items={items} />
          {isImg && <><label>Confidence threshold <b>{conf.toFixed(2)}</b></label><input type="range" min="0.1" max="0.9" step="0.05" value={conf} onChange={(e) => setConf(+e.target.value)} /></>}
          <label>Preview brightness <b>{bright.toFixed(1)}×</b></label><input type="range" min="1" max="4" step="0.1" value={bright} onChange={(e) => setBright(+e.target.value)} />
          <div className="scroll"><table><thead><tr><th>Class</th>{!isImg && <th>First seen</th>}<th>Conf.</th></tr></thead><tbody>
            {items.map((x, i) => <tr key={i}><td><Dot g={x.group} />{x.class_name}</td>{!isImg && <td>{x.first_seen_s}s</td>}<td>{(isImg ? x.confidence : x.max_confidence).toFixed(2)}</td></tr>)}
          </tbody></table></div>
          <a className="btn" href={src} download>{isImg ? "Download original image" : "Download video"}</a>
          {items.length > 0 && <button className="btn alt" onClick={() => saveCsv(isImg ? "detections.csv" : "tracked_objects.csv",
            isImg ? items.map((x) => ({ class: x.class_name, group: x.group, confidence: x.confidence, x1: x.box[0], y1: x.box[1], x2: x.box[2], y2: x.box[3] })) : items)}>Download CSV</button>}
        </div></div></div>);
  }

  return (<div className="panel">
    {err && <p className="err">{err}</p>}
    {runs === null ? <p className="note">Loading…</p> : !runs.length
      ? <p className="note">No saved runs yet. Run a detection and it will appear here.</p>
      : <div className="scroll" style={{ maxHeight: "none" }}><table><thead><tr><th>When</th><th>Type</th><th>File</th><th>Waste</th><th>Animals</th><th>Other</th><th></th></tr></thead><tbody>
        {runs.map((r) => <tr key={r.id}><td>{fmt(r.created)}</td><td>{r.kind}</td><td>{r.name}</td><td>{r.waste}</td><td>{r.animal}</td><td>{r.other}</td>
          <td style={{ whiteSpace: "nowrap" }}><button className="btn" style={{ margin: "0 6px 0 0", padding: "3px 10px" }} onClick={() => view(r.id)}>Open</button>
            <button className="btn alt" style={{ margin: 0, padding: "3px 10px" }} onClick={() => del(r.id)}>Delete</button></td></tr>)}
      </tbody></table></div>}
    <p className="note">Image counts use a 0.25 threshold. Files are stored in backend/storage.</p>
  </div>);
}