from pyspark.sql import SparkSession
from const import dataset_path
import glob
import os


def main():
    parquet_path = f"{dataset_path}/parquet_output"

    subdirectories = [
        os.path.join(parquet_path, d)
        for d in os.listdir(parquet_path)
        if os.path.isdir(os.path.join(parquet_path, d))
    ]

    if not subdirectories:
        print("No parquet subdirectories found in the folder.")
    else:
        spark = (
            SparkSession.builder.appName("SemanticProcessor")
            .config("spark.driver.memory", "8g")
            .config("spark.executor.memory", "8g")
            .config("spark.memory.offHeap.enabled", "true")
            .config("spark.memory.offHeap.size", "8g")
            .getOrCreate()
        )

        for subdir in subdirectories:
            print(f"Processing folder: {subdir}")

            # Get all parquet files within the current subdirectory
            parquet_files = glob.glob(os.path.join(subdir, "*.parquet"))

            if not parquet_files:
                print(f"No parquet files found for {subdir}. Skipping...")
                continue

            df = spark.read.parquet(*parquet_files)

            df.show(5)
            print(f"Finished processing {subdir}\n")

        spark.stop()


if __name__ == "__main__":
    main()
