import pandas as pd

d = pd.read_csv("compare/rtdetr_l_darkaware.csv")
if "subset" in d.columns:
    d["condition"] = d["subset"].str.replace("test_", "", regex=False)
    d["mAP50_95"] = d["mAP50-95"]
    d["f1_score"] = 2 * d["precision"] * d["recall"] / (d["precision"] + d["recall"] + 1e-9)
    d = d[["condition", "precision", "recall", "f1_score", "mAP50", "mAP50_95"]]
d.to_csv("compare/rtdetr_l_darkaware.csv", index=False)
print(d.round(3))