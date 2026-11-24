from pyspark.sql import SparkSession

spark = (
    SparkSession.builder.appName("iceberg-test")
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
spark.sql("DROP TABLE IF EXISTS local.db.t1")

spark.sql("""
    CREATE TABLE local.db.t1 (id BIGINT, value DOUBLE)
    USING iceberg
    TBLPROPERTIES ('write.delete.mode'='copy-on-write',
                   'write.update.mode'='copy-on-write',
                   'write.merge.mode'='copy-on-write')
""")

for i in range(3):
    spark.sql(f"INSERT INTO local.db.t1 VALUES ({i}, {i * 1.5})")

print("---- DATA FILES IN THE TABLE ----")
spark.sql("""SELECT file_path, record_count, file_size_in_bytes
             FROM local.db.t1.files""").show(truncate=False)

spark.stop()
