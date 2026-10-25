import pandas as pd, glob, os
import matplotlib.pyplot as plt

names = {"yolo11l": "YOLO11-l", "yolo26l": "YOLO26-l", "rtdetr_l": "RT-DETR-l"}
labels = {"bright": "bright\n(0)", "dim": "dim\n(0.3)",
          "dark": "dark\n(0.5)", "very_dark": "very dark\n(0.65)"}
order = ["bright", "dim", "dark", "very_dark"]

# 1) Read all files and take the model name from the file name
rows = []
for f in sorted(glob.glob("compare_full/*.csv")):
    name = os.path.basename(f).replace(".csv", "")
    model, training = name.rsplit("_", 1)
    d = pd.read_csv(f)
    d["model"] = model
    d["training"] = training
    rows.append(d)
df = pd.concat(rows, ignore_index=True)

df["condition"] = pd.Categorical(df["condition"], order, ordered=True)
df = df.sort_values(["model", "condition"])
df.to_csv("all_results_full.csv", index=False)
print(df.round(3).to_string(index=False))

# 2) Compact tables for the report
metrics = ["mAP50", "mAP50_95", "trash_only_mAP50_95", "f1_score"]
for m in metrics:
    if m in df.columns:
        t = df.pivot_table(index="model", columns="condition",
                           values=m, observed=True).round(3)
        t.to_csv(f"table_full_{m}.csv")
        print(f"\n{m}\n", t)

# 3) Speed table (average over the four conditions)
if "inference_ms" in df.columns:
    speed = df.groupby("model")["inference_ms"].mean().round(1)
    speed.to_csv("table_full_speed.csv")
    print("\nInference speed (ms per image):\n", speed)

# 4) Plots: one line per model across darkness levels
plots = [("mAP50", "map50_full.png", "mAP50 (all 22 classes)"),
         ("mAP50_95", "map50_95_full.png", "mAP50-95 (all 22 classes)"),
         ("trash_only_mAP50_95", "trash_only_map50_95_full.png", "mAP50-95 (14 trash classes only)")]

for metric, fname, title in plots:
    if metric not in df.columns:
        continue
    fig, ax = plt.subplots(figsize=(6, 4))
    for model, g in df.groupby("model"):
        x = [labels[c] for c in g["condition"].astype(str)]
        ax.plot(x, g[metric], marker="o", label=names.get(model, model))
    ax.set_title(title)
    ax.set_xlabel("Darkness level")
    ax.set_ylabel(metric.replace("_", "-"))
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(fname, dpi=200)
    plt.close()

print("\nSaved: all_results_full.csv, table_full_*.csv, and the three plots (*_full.png)")