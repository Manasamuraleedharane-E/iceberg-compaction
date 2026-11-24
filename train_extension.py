"""
Extension model: does the baseline feature set predict how much compaction
speeds up ML-training reads on merge-on-read tables, and do new features help?

Inputs (written by mor_benchmark.py):
  mor_results.csv, data/mor_files_before/*.csv, data/mor_snapshots/*.csv
Outputs:
  mor_dataset.csv and the printed comparison
"""
import glob, os
import numpy as np, pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import RepeatedKFold, cross_val_predict
from sklearn.metrics import r2_score, mean_squared_error

MIN_SZ = 786432

BASE = ["file_count", "total_size_bytes", "avg_file_size_bytes",
        "min_file_size_bytes", "max_file_size_bytes", "stddev_file_size_bytes",
        "total_records", "avg_records_per_file", "num_partitions_actual",
        "avg_files_per_partition", "max_files_per_partition",
        "min_files_per_partition", "stddev_files_per_partition", "num_snapshots",
        "small_file_ratio", "file_size_cv", "files_per_partition_cv"]
NEW = ["delete_file_count", "delete_bytes_ratio", "deleted_row_ratio",
       "delete_files_per_data_file", "commits_since_compaction",
       "upserted_row_ratio_since_compaction"]


def features(table):
    f = pd.read_csv(f"data/mor_files_before/{table}.csv")
    snaps = pd.read_csv(f"data/mor_snapshots/{table}.csv")
    data, dels = f[f["content"] == 0], f[f["content"] != 0]

    # --- the paper's 17 features, from DATA files only (same as the baseline)
    s = data["file_size_bytes"]
    pc = data.groupby("part").size()
    sd = s.std(ddof=1) if len(s) > 1 else 0.0
    pcs = pc.std(ddof=1) if len(pc) > 1 else 0.0
    row = dict(
        table=table, file_count=len(data), total_size_bytes=s.sum(),
        avg_file_size_bytes=s.mean(), min_file_size_bytes=s.min(),
        max_file_size_bytes=s.max(), stddev_file_size_bytes=sd,
        total_records=data["record_count"].sum(),
        avg_records_per_file=data["record_count"].mean(),
        num_partitions_actual=len(pc), avg_files_per_partition=pc.mean(),
        max_files_per_partition=pc.max(), min_files_per_partition=pc.min(),
        stddev_files_per_partition=pcs, num_snapshots=len(snaps),
        small_file_ratio=(s < MIN_SZ).mean(),
        file_size_cv=sd / s.mean() if s.mean() > 0 else 0.0,
        files_per_partition_cv=pcs / pc.mean() if pc.mean() > 0 else 0.0)

    # --- new merge-on-read features: how much delete work a reader must do
    total_rec = max(1, row["total_records"])
    row["delete_file_count"] = len(dels)
    row["delete_bytes_ratio"] = dels["file_size_bytes"].sum() / max(1, s.sum())
    row["deleted_row_ratio"] = dels["record_count"].sum() / total_rec
    row["delete_files_per_data_file"] = len(dels) / max(1, len(data))

    # --- update recency, from snapshot history: commits and upserted rows since
    # the last compaction ('replace' = a rewrite_data_files commit)
    rep = snaps.index[snaps["operation"] == "replace"]
    since = snaps.iloc[rep[-1] + 1:] if len(rep) else snaps
    upserts = since[since["operation"] == "overwrite"]
    row["commits_since_compaction"] = len(since)
    row["upserted_row_ratio_since_compaction"] = (
        upserts["added_records"].sum() / total_rec)
    return row


res = pd.read_csv("mor_results.csv")
feat = pd.DataFrame([features(t) for t in res["table"]
                     if os.path.exists(f"data/mor_files_before/{t}.csv")])
df = feat.merge(res, on="table")
# Targets. Speedup > 1 means compaction made training reads faster.
df["speedup_s1"] = df["read_s_s0"] / df["read_s_s1"]
df["speedup_s2"] = df["read_s_s0"] / df["read_s_s2"]
df["log_speedup_s1"] = np.log(df["speedup_s1"])
df["log_speedup_s2"] = np.log(df["speedup_s2"])
df.to_csv("mor_dataset.csv", index=False)
print(f"{len(df)} tables\n")


def cv_score(cols, target, seed=42):
    """Repeated 5-fold CV: with few tables, one split is too noisy."""
    model = XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05,
                         subsample=0.8, min_child_weight=3, random_state=seed)
    r2s, rmses = [], []
    for rep in range(5):
        cv = RepeatedKFold(n_splits=5, n_repeats=1, random_state=seed + rep)
        p = cross_val_predict(model, df[cols], df[target], cv=cv)
        r2s.append(r2_score(df[target], p))
        rmses.append(mean_squared_error(df[target], p) ** 0.5)
    return np.mean(r2s), np.std(r2s), np.mean(rmses)


print("Does the paper's model still work, and do the new features help?")
for target in ["r_fc", "speedup_s1", "speedup_s2", "log_speedup_s1", "log_speedup_s2"]:
    
    for name, cols in [("baseline 17", BASE), ("17 + new", BASE + NEW)]:
        m, sdv, rmse = cv_score(cols, target)
        print(f"  {target:11s} {name:12s} R2 = {m:.3f} ± {sdv:.3f}   RMSE = {rmse:.3f}")
    print()

print("Does the 'compact if max_files_per_partition > k' rule pick the tables")
print("where compaction actually speeds up training reads (speedup_s1 > 1)?")
faster = (df["speedup_s1"] > 1).astype(int)
print(f"  share of tables where S1 was faster: {faster.mean():.2f}")
for k in range(1, 12):
    acc = ((df["max_files_per_partition"] > k).astype(int) == faster).mean()
    print(f"  > {k}: accuracy {acc:.3f}")

full = XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05,
                    subsample=0.8, min_child_weight=3, random_state=42)
full.fit(df[BASE + NEW], df["speedup_s2"])
imp = pd.Series(full.feature_importances_, index=BASE + NEW)
print("\nTop features for speedup_s2:\n", imp.sort_values(ascending=False).head(8))
