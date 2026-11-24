"""
Extension: merge-on-read (MoR) tables + ML-training-style read benchmark.

For every table in the grid this script:
  1. builds an Iceberg table the same way generate_tables.py does, but with an
     `id` key and merge-on-read write modes,
  2. applies rounds of MERGE INTO upserts (this creates position delete files),
     optionally with one compaction in the middle (so "update recency" varies),
  3. saves the metadata it needs for features (file list incl. delete files,
     snapshot history) BEFORE any final compaction,
  4. measures, in three compaction states, the training-style full-scan read
     time, storage size and the time of one small upsert:
        S0 = no compaction (fragmented, with delete files)
        S1 = the paper's bin-pack compaction (same scaled settings as baseline)
        S2 = full maintenance (rewrite every data file + rewrite delete files)
  5. appends one row per table to mor_results.csv (safe to stop and re-run).

Run a quick test first with MAX_TABLES = 2.
"""
import csv, os, random, statistics, time, itertools
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

MAX_TABLES = 96           # test with 2, then raise to len(grid)
RUNS = 3                 # timed runs per read (after 1 warm-up), median reported
TARGET = 1048576         # same scaled-down sizes as generate_tables.py
MIN_SZ = 786432
MAX_SZ = 1887437
PROBE_FRAC = 0.01        # size of the upsert used to measure update latency
OUT = "mor_results.csv"
NS = "mor"               # separate namespace, so the baseline tables are untouched
# Warehouse folder: set the ICEBERG_WAREHOUSE environment variable to override.
WAREHOUSE = os.environ.get("ICEBERG_WAREHOUSE", "D:/iceberg_warehouse")

spark = (
    SparkSession.builder.appName("mor-benchmark")
    .master("local[4]")  # fixed core count so timings are comparable between runs
    .config("spark.jars.packages",
            "org.apache.iceberg:iceberg-spark-runtime-3.5_2.12:1.5.2")
    .config("spark.sql.extensions",
            "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions")
    .config("spark.sql.catalog.local", "org.apache.iceberg.spark.SparkCatalog")
    .config("spark.sql.catalog.local.type", "hadoop")
    .config("spark.sql.catalog.local.warehouse", WAREHOUSE)
    .config("spark.ui.enabled", "false")
    .config("spark.sql.shuffle.partitions", "4")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("ERROR")
spark.sql(f"CREATE NAMESPACE IF NOT EXISTS local.{NS}")
os.makedirs("data/mor_files_before", exist_ok=True)
os.makedirs("data/mor_snapshots", exist_ok=True)


# ---------------------------------------------------------------- data makers
def part_col(n_parts, skew):
    """Partition number from a uniform random u (same logic as the baseline)."""
    u = F.col("u")
    if n_parts == 1 or skew == "uniform":
        return F.floor(u * n_parts).cast("int")
    w = [1.0 / (k + 1) for k in range(n_parts)]
    cum, acc = [], 0.0
    for x in w:
        acc += x / sum(w)
        cum.append(acc)
    expr = F.lit(n_parts - 1)
    for k in range(n_parts - 2, -1, -1):
        expr = F.when(u < cum[k], F.lit(k)).otherwise(expr)
    return expr.cast("int")


def new_rows(start_id, n, n_cols, n_parts, skew):
    """n brand-new rows with ids start_id .. start_id+n-1."""
    df = spark.range(start_id, start_id + n).withColumn("u", F.rand())
    df = df.withColumn("part", part_col(n_parts, skew))
    cols = [F.expr("uuid()").alias(f"c{i}") for i in range(n_cols)]
    return df.select("id", "part", *cols)


def upsert(full, frac, next_id, n_cols, n_parts, skew):
    """One MERGE INTO: update `frac` of existing rows, insert frac/5 new rows.
    Updated rows keep their id and partition; only the payload changes.
    Returns the next unused id."""
    cols = [F.expr("uuid()").alias(f"c{i}") for i in range(n_cols)]
    upd = spark.table(full).sample(fraction=frac).select("id", "part", *cols)
    n_ins = max(1, int(upd.count() / 5))
    src = upd.unionByName(new_rows(next_id, n_ins, n_cols, n_parts, skew))
    src.createOrReplaceTempView("src")
    spark.sql(f"""MERGE INTO {full} t USING src s ON t.id = s.id
                  WHEN MATCHED THEN UPDATE SET *
                  WHEN NOT MATCHED THEN INSERT *""")
    return next_id + n_ins


def binpack(tname, extra=""):
    """The paper's bin-pack compaction with the scaled-down baseline sizes."""
    opts = (f"map('target-file-size-bytes','{TARGET}',"
            f"'min-file-size-bytes','{MIN_SZ}',"
            f"'max-file-size-bytes','{MAX_SZ}'{extra})")
    spark.sql(f"CALL local.system.rewrite_data_files(table => '{NS}.{tname}', "
              f"strategy => 'binpack', options => {opts})").collect()


# --------------------------------------------------------------- measurements
def current_snapshot(full):
    return spark.sql(f"SELECT snapshot_id FROM {full}.snapshots "
                     f"ORDER BY committed_at DESC LIMIT 1").first()[0]


def storage(full):
    """Live data files / delete files and their bytes in the current state.
    content: 0 = data file, 1 = position deletes, 2 = equality deletes."""
    rows = spark.sql(f"SELECT content, count(*) n, sum(file_size_in_bytes) b "
                     f"FROM {full}.files GROUP BY content").collect()
    d = {r["content"]: (r["n"], r["b"]) for r in rows}
    data_n, data_b = d.get(0, (0, 0))
    del_n = sum(v[0] for k, v in d.items() if k != 0)
    del_b = sum(v[1] for k, v in d.items() if k != 0)
    return data_n, del_n, data_b + del_b


def training_read(full, snap):
    """Training-style read: every row, every column, deletes applied.
    The 'noop' sink makes Spark read and decode everything but write nothing,
    so the time is pure read cost. Also returns the number of read tasks."""
    df = spark.sql(f"SELECT * FROM {full} VERSION AS OF {snap}")
    tasks = df.rdd.getNumPartitions()
    times = []
    for i in range(1 + RUNS):
        t = time.perf_counter()
        df.write.format("noop").mode("overwrite").save()
        times.append(time.perf_counter() - t)
    return round(statistics.median(times[1:]), 4), tasks


def update_latency(full, tname, next_id, n_cols, n_parts, skew):
    """Time one small upsert in the current state, then roll it back so the
    table is exactly as before."""
    snap = current_snapshot(full)
    t = time.perf_counter()
    upsert(full, PROBE_FRAC, next_id, n_cols, n_parts, skew)
    secs = time.perf_counter() - t
    spark.sql(f"CALL local.system.rollback_to_snapshot('{NS}.{tname}', {snap})")
    return round(secs, 4)


def measure(full, tname, next_id, n_cols, n_parts, skew):
    snap = current_snapshot(full)
    read_s, tasks = training_read(full, snap)
    data_n, del_n, total_b = storage(full)
    upd_s = update_latency(full, tname, next_id, n_cols, n_parts, skew)
    return dict(read_s=read_s, tasks=tasks, data_files=data_n,
                delete_files=del_n, bytes=total_b, update_s=upd_s)


def save_metadata(full, tname):
    """Feature inputs, taken from Iceberg metadata only."""
    files = spark.sql(f"SELECT content, partition, file_size_in_bytes, "
                      f"record_count FROM {full}.files").collect()
    with open(f"data/mor_files_before/{tname}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["content", "part", "file_size_bytes", "record_count"])
        for r in files:
            w.writerow([r["content"], r["partition"]["part"],
                        r["file_size_in_bytes"], r["record_count"]])
    snaps = spark.sql(f"SELECT committed_at, operation, summary "
                      f"FROM {full}.snapshots ORDER BY committed_at").collect()
    with open(f"data/mor_snapshots/{tname}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["committed_at", "operation", "added_records",
                    "added_delete_files", "added_position_deletes"])
        for r in snaps:
            s = r["summary"] or {}
            w.writerow([r["committed_at"], r["operation"],
                        s.get("added-records", 0), s.get("added-delete-files", 0),
                        s.get("added-position-deletes", 0)])


# ----------------------------------------------------------------- the grid
grid = []
for rows, cols, parts, batches, rounds, frac, mid in itertools.product(
        [10000, 50000], [5, 20], [1, 5], [3, 8], [2, 5, 10], [0.02, 0.10],
        [False, True]):
    grid.append((rows, cols, parts, batches, rounds, frac, mid))
random.Random(42).shuffle(grid)
numbered = list(enumerate(grid, start=1))[:MAX_TABLES]

header = ["table", "num_rows", "num_columns", "num_partitions",
          "num_write_batches", "upsert_rounds", "upsert_frac", "mid_compaction",
          "n_before", "n_after", "r_fc"] + [
          f"{k}_{s}" for s in ("s0", "s1", "s2")
          for k in ("read_s", "tasks", "data_files", "delete_files", "bytes",
                    "update_s")] + ["seconds"]
done = set()
if os.path.exists(OUT):
    with open(OUT, newline="") as f:
        done = {r["table"] for r in csv.DictReader(f)}
else:
    with open(OUT, "w", newline="") as f:
        csv.writer(f).writerow(header)


def run_one(idx, rows, cols, parts, batches, rounds, frac, mid):
    tname, skew = f"m_{idx:03d}", "uniform"
    full = f"local.{NS}.{tname}"
    t0 = time.time()
    spark.sql(f"DROP TABLE IF EXISTS {full} PURGE")
    col_sql = ", ".join(["id BIGINT", "part INT"] +
                        [f"c{i} STRING" for i in range(cols)])
    props = ("'format-version'='2',"
             "'write.delete.mode'='merge-on-read',"
             "'write.update.mode'='merge-on-read',"
             "'write.merge.mode'='merge-on-read',"
             "'write.distribution-mode'='none',"
             "'write.spark.fanout.enabled'='true',"
             f"'write.target-file-size-bytes'='{TARGET}',"
             # Iceberg packs small files into read splits of this size (default
             # 128 MB). Scale it down like the file sizes, so the number of read
             # tasks behaves like it would on full-size tables.
             f"'read.split.target-size'='{TARGET}'")
    spark.sql(f"CREATE TABLE {full} ({col_sql}) USING iceberg "
              f"PARTITIONED BY (part) TBLPROPERTIES ({props})")

    # appends: first batch 50 % of rows, the rest share the other 50 %
    first = rows // 2
    rest = max(1, (rows - first) // (batches - 1))
    next_id = 0
    for n in [first] + [rest] * (batches - 1):
        new_rows(next_id, n, cols, parts, skew).writeTo(full).append()
        next_id += n

    # upserts, with an optional compaction halfway (changes update recency)
    for r in range(rounds):
        if mid and r == rounds // 2:
            binpack(tname)
        next_id = upsert(full, frac, next_id, cols, parts, skew)

    save_metadata(full, tname)
    out = {}
    out["s0"] = measure(full, tname, next_id, cols, parts, skew)
    binpack(tname)                                        # S1: paper's compaction
    out["s1"] = measure(full, tname, next_id, cols, parts, skew)
    binpack(tname, ",'rewrite-all','true'")               # S2: full maintenance
    spark.sql(f"CALL local.system.rewrite_position_delete_files("
              f"table => '{NS}.{tname}')").collect()
    out["s2"] = measure(full, tname, next_id, cols, parts, skew)

    n_before, n_after = out["s0"]["data_files"], out["s1"]["data_files"]
    row = [tname, rows, cols, parts, batches, rounds, frac, int(mid),
           n_before, n_after, round(1 - n_after / n_before, 4)]
    for s in ("s0", "s1", "s2"):
        row += [out[s][k] for k in ("read_s", "tasks", "data_files",
                                    "delete_files", "bytes", "update_s")]
    return row + [round(time.time() - t0, 1)]


for idx, cfg in numbered:
    tname = f"m_{idx:03d}"
    if tname in done:
        print(f"[{idx}] {tname} already done, skipping", flush=True)
        continue
    try:
        row = run_one(idx, *cfg)
        with open(OUT, "a", newline="") as f:
            csv.writer(f).writerow(row)
        print(f"[{idx}] {tname} r_fc={row[10]} read s0/s1/s2 = "
              f"{row[11]}/{row[17]}/{row[23]} s  ({row[-1]}s)", flush=True)
    except Exception as e:
        print(f"[{idx}] {tname} FAILED: {str(e)[:300]}", flush=True)

print("ALL DONE")
spark.stop()
