import pandas as pd

d = pd.read_csv("per_class/yolo26l_darkaware_per_class.csv")

# one row per class, one column per darkness level
t = d.pivot_table(index="class", columns="condition", values="mAP50_95")
t = t[["bright", "dim", "dark", "very_dark"]]
t["drop_bright_to_very_dark"] = t["bright"] - t["very_dark"]
t = t.sort_values("very_dark")

print(t.round(3).to_string())
t.round(3).to_csv("per_class_table.csv")
print("\nWeakest 3 classes on very_dark:", list(t.index[:3]))