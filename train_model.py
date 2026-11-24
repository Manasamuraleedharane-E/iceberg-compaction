import numpy as np, pandas as pd
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

FEATURES = ["file_count","total_size_bytes","avg_file_size_bytes","min_file_size_bytes",
 "max_file_size_bytes","stddev_file_size_bytes","total_records","avg_records_per_file",
 "num_partitions_actual","avg_files_per_partition","max_files_per_partition",
 "min_files_per_partition","stddev_files_per_partition","num_snapshots",
 "small_file_ratio","file_size_cv","files_per_partition_cv"]

df = pd.read_csv("dataset.csv")
X, y, yc = df[FEATURES], df["file_reduction_ratio"], df["needs_compaction"]
strat = yc if yc.value_counts().min() >= 2 else None
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=strat)

xgb = XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, subsample=0.8,
                   colsample_bytree=0.8, min_child_weight=3, random_state=42)
for name, m in {"XGBoost": xgb, "Linear": LinearRegression()}.items():
    m.fit(Xtr, ytr); p = m.predict(Xte)
    cv = cross_val_score(m, X, y, cv=KFold(5, shuffle=True, random_state=42), scoring="r2")
    print(f"{name}: R2={r2_score(yte,p):.3f} RMSE={mean_squared_error(yte,p)**0.5:.3f} "
          f"MAE={mean_absolute_error(yte,p):.3f} | 5-fold R2={cv.mean():.3f}")

xgb.fit(X, y)
imp = pd.Series(xgb.feature_importances_, index=FEATURES).sort_values(ascending=False)
print("\nTop features:\n", imp.head(10))

print("\nThreshold sweep on max_files_per_partition:")
for k in range(1, 12):
    acc = ((df["max_files_per_partition"] > k).astype(int) == yc).mean()
    print(f"  > {k}: accuracy {acc:.3f}")