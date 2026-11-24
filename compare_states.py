import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("mor_results.csv")
states = [("s0", "S0 none"), ("s1", "S1 bin-pack"), ("s2", "S2 full")]
rows = []
for s, label in states:
    rows.append({"state": label,
                 "read_s": df[f"read_s_{s}"].median(),
                 "read_tasks": df[f"tasks_{s}"].median(),
                 "data_files": df[f"data_files_{s}"].median(),
                 "delete_files": df[f"delete_files_{s}"].median(),
                 "storage_MB": df[f"bytes_{s}"].median() / 1e6,
                 "update_s": df[f"update_s_{s}"].median()})
summary = pd.DataFrame(rows)
print(summary.round(3).to_string(index=False))
summary.to_csv("extension_summary.csv", index=False)

for s in ["s1", "s2"]:
    sp = df["read_s_s0"] / df[f"read_s_{s}"]
    print(f"read speedup {s}: median {sp.median():.2f}x, mean {sp.mean():.2f}x, "
          f"max {sp.max():.1f}x, faster in {(sp > 1).mean():.0%} of tables")

fig, ax = plt.subplots(1, 3, figsize=(12, 3.5))
for a, col, title in zip(ax, ["read_s", "storage_MB", "update_s"],
                         ["Training read time (s)", "Storage (MB)", "Update latency (s)"]):
    a.bar(summary["state"], summary[col]); a.set_title(title + ", median")
plt.tight_layout(); plt.savefig("figures/extension_states.png", dpi=200)

d = pd.read_csv("mor_dataset.csv")
cols = ["speedup_s1", "delete_file_count", "delete_files_per_data_file",
        "file_count", "commits_since_compaction"]
print(d[cols].corr(method="spearman")["speedup_s1"].round(2))