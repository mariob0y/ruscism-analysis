from pyspark.sql import SparkSession
from pyspark.sql.functions import explode
from const import dataset_path
import glob
import os


def main():
    datasets = glob.glob(os.path.join(dataset_path, "*.json"))

    if not datasets:
        print("No datasets found in the folder.")
        return

    spark = (
        SparkSession.builder.appName("Json2Parquet")
        .config("spark.driver.memory", "8g")
        .config("spark.executor.memory", "8g")
        .config("spark.memory.offHeap.enabled", "true")
        .config("spark.memory.offHeap.size", "8g")
        .getOrCreate()
    )

    for file_path in datasets:
        dataset_name = os.path.splitext(os.path.basename(file_path))[0]
        output_path = f"{dataset_path}/parquet/{dataset_name}"

        if os.path.exists(output_path):
            continue

        print(f"Processing {dataset_name}.json")

        df = spark.read.option("multiLine", "true").json(file_path)
        df_exploded = df.select(explode("messages").alias("message"))
        df_final = df_exploded.select("message.*")
        df_final.write.option("maxRecordsPerFile", 1_000_00).parquet(output_path)


if __name__ == "__main__":
    main()
