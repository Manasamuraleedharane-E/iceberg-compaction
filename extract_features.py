import glob, os
import pandas as pd

MIN_SZ = 786432
gt = pd.read_csv("ground_truth.csv")
rows = []
for path in sorted(glob.glob("data/files_before/*.csv")):
    t = os.path.basename(path)[:-4]
    f = pd.read_csv(path)
    s = f["file_size_bytes"]
    pc = f.groupby("part").size()
    sd = s.std(ddof=1) if len(s) > 1 else 0.0
    pcs = pc.std(ddof=1) if len(pc) > 1 else 0.0
    rows.append(dict(
        table=t, file_count=len(f), total_size_bytes=s.sum(),
        avg_file_size_bytes=s.mean(), min_file_size_bytes=s.min(),
        max_file_size_bytes=s.max(), stddev_file_size_bytes=sd,
        total_records=f["record_count"].sum(),
        avg_records_per_file=f["record_count"].mean(),
        num_partitions_actual=len(pc),
        avg_files_per_partition=pc.mean(), max_files_per_partition=pc.max(),
        min_files_per_partition=pc.min(), stddev_files_per_partition=pcs,
        small_file_ratio=(s < MIN_SZ).mean(),
        file_size_cv=sd / s.mean() if s.mean() > 0 else 0.0,
        files_per_partition_cv=pcs / pc.mean() if pc.mean() > 0 else 0.0,
    ))
feat = pd.DataFrame(rows).merge(
    gt[["table", "num_snapshots", "n_before", "n_after", "r_fc"]], on="table")
feat = feat.rename(columns={"r_fc": "file_reduction_ratio"})
feat["needs_compaction"] = (feat["file_reduction_ratio"] > 0).astype(int)
feat.to_csv("dataset.csv", index=False)
print(feat.shape)
print(feat["needs_compaction"].value_counts())