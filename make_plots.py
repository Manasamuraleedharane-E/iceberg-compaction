import os
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

FEATURES = ["file_count","total_size_bytes","avg_file_size_bytes","min_file_size_bytes",
 "max_file_size_bytes","stddev_file_size_bytes","total_records","avg_records_per_file",
 "num_partitions_actual","avg_files_per_partition","max_files_per_partition",
 "min_files_per_partition","stddev_files_per_partition","num_snapshots",
 "small_file_ratio","file_size_cv","files_per_partition_cv"]

os.makedirs("figures", exist_ok=True)
df = pd.read_csv("dataset.csv")
X, y = df[FEATURES], df["file_reduction_ratio"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
m = XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, subsample=0.8,
                 colsample_bytree=0.8, min_child_weight=3, random_state=42).fit(Xtr, ytr)

# 1. predicted vs actual
plt.figure(figsize=(5, 5))
plt.scatter(yte, m.predict(Xte), alpha=0.7)
plt.plot([0, 1], [0, 1], "--", color="gray")
plt.xlabel("Actual file reduction ratio"); plt.ylabel("Predicted")
plt.tight_layout(); plt.savefig("figures/pred_vs_actual.png", dpi=200); plt.close()

# 2. feature importance
imp = pd.Series(m.fit(X, y).feature_importances_, index=FEATURES).sort_values().tail(10)
plt.figure(figsize=(6, 4)); imp.plot.barh()
plt.xlabel("Importance (gain)"); plt.tight_layout()
plt.savefig("figures/feature_importance.png", dpi=200); plt.close()

# 3. threshold sweep
ks = list(range(1, 12))
acc = [((df["max_files_per_partition"] > k).astype(int) == df["needs_compaction"]).mean() for k in ks]
plt.figure(figsize=(5, 3.5)); plt.plot(ks, acc, marker="o")
plt.xlabel("k  (max_files_per_partition > k)"); plt.ylabel("Accuracy")
plt.tight_layout(); plt.savefig("figures/threshold_sweep.png", dpi=200); plt.close()
print("saved to figures/")