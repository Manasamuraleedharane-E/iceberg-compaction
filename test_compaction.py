from pyspark.sql import SparkSession

spark = (
    SparkSession.builder.appName("compaction-test")
    .config("spark.jars.packages",
            "org.apache.iceberg:iceberg-spark-runtime-3.5_2.12:1.5.2")
    .config("spark.sql.extensions",
            "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions")
    .config("spark.sql.catalog.local", "org.apache.iceberg.spark.SparkCatalog")
    .config("spark.sql.catalog.local.type", "hadoop")
    .config("spark.sql.catalog.local.warehouse", "D:/iceberg_warehouse")
    .getOrCreate()
)

spark.sql("CREATE NAMESPACE IF NOT EXISTS local.db")
spark.sql("DROP TABLE IF EXISTS local.db.t2")

spark.sql("""
    CREATE TABLE local.db.t2 (id BIGINT, value DOUBLE)
    USING iceberg
    TBLPROPERTIES ('write.delete.mode'='copy-on-write',
                   'write.update.mode'='copy-on-write',
                   'write.merge.mode'='copy-on-write')
""")

# 6 separate inserts = 6 small files
for i in range(6):
    spark.sql(f"INSERT INTO local.db.t2 VALUES ({i}, {i * 1.5})")

n_before = spark.sql("SELECT * FROM local.db.t2.files").count()

# Paper's settings: target 128 MB, min 96 MB, max about 230 MB
spark.sql("""
    CALL local.system.rewrite_data_files(
        table => 'db.t2',
        strategy => 'binpack',
        options => map(
            'target-file-size-bytes', '134217728',
            'min-file-size-bytes',    '100663296',
            'max-file-size-bytes',    '241591910'
        )
    )
""").show(truncate=False)

n_after = spark.sql("SELECT * FROM local.db.t2.files").count()

print("files before:", n_before)
print("files after :", n_after)
print("r_fc = 1 - after/before =", round(1 - n_after / n_before, 3))

spark.stop()
