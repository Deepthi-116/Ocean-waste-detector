import pandas as pd, glob, os
import matplotlib.pyplot as plt

rows = []
for f in glob.glob("compare/*.csv"):
    name = os.path.basename(f).replace(".csv", "")
    model, training = name.rsplit("_", 1)
    d = pd.read_csv(f); d["model"] = model; d["training"] = training
    rows.append(d)
df = pd.concat(rows)

order = ["bright", "dim", "dark", "very_dark"]
df["condition"] = pd.Categorical(df["condition"], order, ordered=True)
df = df.sort_values(["training", "model", "condition"])
df.to_csv("all_results.csv", index=False)
print(df.round(3).to_string(index=False))

# one plot per training type: mAP50 vs darkness
fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for ax, tr in zip(axes, ["baseline", "darkaware"]):
    for model, g in df[df.training == tr].groupby("model"):
        ax.plot(g["condition"].astype(str), g["mAP50"], marker="o", label=model)
    ax.set_title(f"{tr} training"); ax.set_xlabel("darkness"); ax.grid(alpha=.3)
axes[0].set_ylabel("mAP50"); axes[0].legend()
plt.tight_layout(); plt.savefig("map_vs_darkness.png", dpi=200)