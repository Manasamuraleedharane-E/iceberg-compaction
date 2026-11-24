import csv, os, random, time, itertools
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

MAX_TABLES = 72         # start small. Later change to 72
TARGET = 1048576         # 1 MB     (paper: 128 MB)
MIN_SZ = 786432          # 0.75 MB  (paper: 96 MB)
MAX_SZ = 1887437         # 1.8 MB   (paper: about 230 MB)
GT = "ground_truth.csv"

spark = (
    SparkSession.builder.appName("generate-tables")
    .config("spark.jars.packages",
            "org.apache.iceberg:iceberg-spark-runtime-3.5_2.12:1.5.2")
    .config("spark.sql.extensions",
            "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions")
    .config("spark.sql.catalog.local", "org.apache.iceberg.spark.SparkCatalog")
    .config("spark.sql.catalog.local.type", "hadoop")
    .config("spark.sql.catalog.local.warehouse", "D:/iceberg_warehouse")
    .config("spark.ui.enabled", "false")
    .config("spark.sql.shuffle.partitions", "4")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("ERROR")
spark.sql("CREATE NAMESPACE IF NOT EXISTS local.sim")
os.makedirs("data/files_before", exist_ok=True)


def part_expr(n_parts, skew):
    u = F.col("u")
    if n_parts == 1 or skew == "uniform":
        return F.floor(u * n_parts).cast("int")
    w = [1.0 / (k + 1) for k in range(n_parts)]
    total = sum(w)
    cum, acc = [], 0.0
    for x in w:
        acc += x / total
        cum.append(acc)
    expr = F.lit(n_parts - 1)
    for k in range(n_parts - 2, -1, -1):
        expr = F.when(u < cum[k], F.lit(k)).otherwise(expr)
    return expr.cast("int")


def make_batch(n_rows, n_cols, n_parts, skew):
    df = spark.range(n_rows).withColumn("u", F.rand())
    df = df.withColumn("part", part_expr(n_parts, skew))
    cols = [F.expr("uuid()").alias(f"c{i}") for i in range(n_cols)]
    return df.select(F.col("part"), *cols)

# ---- the grid of table settings ----
grid = []
for rows, cols, parts, writers, batches, skew in itertools.product(
        [10000, 50000], [5, 20], [1, 5], [1, 3], [3, 8, 15],
        ["uniform", "zipf"]):
    if parts == 1 and skew == "zipf":
        continue
    grid.append((rows, cols, parts, writers, batches, skew))
random.Random(42).shuffle(grid)
numbered = list(enumerate(grid, start=1))[:MAX_TABLES]

# ---- skip tables that are already done, so you can re-run safely ----
done = set()
if os.path.exists(GT):
    with open(GT, newline="") as f:
        done = {r["table"] for r in csv.DictReader(f)}

header = ["table", "num_rows", "num_columns", "num_partitions", "num_writers",
          "num_write_batches", "skew", "n_before", "n_after", "r_fc",
          "num_snapshots", "seconds"]
if not os.path.exists(GT):
    with open(GT, "w", newline="") as f:
        csv.writer(f).writerow(header)

def build_and_compact(idx, rows, cols, parts, writers, batches, skew):
    tname = f"t_{idx:03d}"
    full = f"local.sim.{tname}"
    t0 = time.time()
    spark.sql(f"DROP TABLE IF EXISTS {full} PURGE")

    col_sql = ", ".join(["part INT"] + [f"c{i} STRING" for i in range(cols)])
    props = ("'write.delete.mode'='copy-on-write',"
             "'write.update.mode'='copy-on-write',"
             "'write.merge.mode'='copy-on-write',"
             "'write.distribution-mode'='none',"
             "'write.spark.fanout.enabled'='true',"  "'write.target-file-size-bytes'='1048576'")
    spark.sql(f"CREATE TABLE {full} ({col_sql}) USING iceberg "
              f"PARTITIONED BY (part) TBLPROPERTIES ({props})")

    # first batch gets 50% of rows, the rest share the other 50%
    first = rows // 2
    rest = max(1, (rows - first) // (batches - 1))
    sizes = [first] + [rest] * (batches - 1)
    for n in sizes:
        make_batch(n, cols, parts, skew).repartition(writers) \
            .writeTo(full).append()

    # save the file list BEFORE compaction (used for features in Step 5)
    files = spark.sql(f"SELECT * FROM {full}.files").collect()
    n_before = len(files)
    n_snap = spark.sql(f"SELECT * FROM {full}.snapshots").count()
    with open(f"data/files_before/{tname}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["part", "file_size_bytes", "record_count"])
        for r in files:
            w.writerow([r["partition"]["part"], r["file_size_in_bytes"],
                        r["record_count"]])

    # compaction with the paper's settings (scaled down)
    opts = (f"map('target-file-size-bytes','{TARGET}',"
            f"'min-file-size-bytes','{MIN_SZ}',"
            f"'max-file-size-bytes','{MAX_SZ}')")
    spark.sql(f"CALL local.system.rewrite_data_files(table => 'sim.{tname}', "
              f"strategy => 'binpack', options => {opts})").collect()

    n_after = spark.sql(f"SELECT * FROM {full}.files").count()
    r_fc = round(1 - n_after / n_before, 4)
    secs = round(time.time() - t0, 1)
    return [tname, rows, cols, parts, writers, batches, skew,
            n_before, n_after, r_fc, n_snap, secs]


# ---- main loop: build every table and record the result ----
for idx, (rows, cols, parts, writers, batches, skew) in numbered:
    tname = f"t_{idx:03d}"
    if tname in done:
        print(f"[{idx}] {tname} already done, skipping", flush=True)
        continue
    try:
        row = build_and_compact(idx, rows, cols, parts, writers, batches, skew)
        with open(GT, "a", newline="") as f:
            csv.writer(f).writerow(row)
        print(f"[{idx}] {tname}  before={row[7]}  after={row[8]}  "
              f"r_fc={row[9]}  ({row[11]}s)", flush=True)
    except Exception as e:
        print(f"[{idx}] {tname} FAILED: {str(e)[:300]}", flush=True)

print("ALL DONE")
spark.stop()

