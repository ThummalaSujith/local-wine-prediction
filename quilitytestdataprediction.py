import os
import sys
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.types import StructType, StructField, DoubleType, StringType

import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    try:
        # Check arguments
        if len(sys.argv) < 3:
            logger.error("Usage: python test_model.py <model_dir> <local_test_dataset_path>")
            sys.exit(1)

        # Arguments
        model_dir = sys.argv[1]
        local_test_dataset_path = sys.argv[2]

        # Start Spark session
        logger.info("Starting Spark session...")
        spark = SparkSession.builder \
            .appName('Test_Wine_Quality_Model') \
            .config("spark.sql.shuffle.partitions", "50") \
            .getOrCreate()
        spark.sparkContext.setLogLevel('ERROR')

        # Define schema for test dataset
        schema = StructType([
            StructField("fixed acidity", DoubleType(), True),
            StructField("volatile acidity", DoubleType(), True),
            StructField("citric acid", DoubleType(), True),
            StructField("residual sugar", DoubleType(), True),
            StructField("chlorides", DoubleType(), True),
            StructField("free sulfur dioxide", DoubleType(), True),
            StructField("total sulfur dioxide", DoubleType(), True),
            StructField("density", DoubleType(), True),
            StructField("pH", DoubleType(), True),
            StructField("sulphates", DoubleType(), True),
            StructField("alcohol", DoubleType(), True),
            StructField("quality", StringType(), True)
        ])

        # Load test dataset
        logger.info("Loading test dataset...")
        test_df = spark.read.csv(local_test_dataset_path, schema=schema, header=True, sep=";")

        # Load trained model
        logger.info(f"Loading model from {model_dir}...")
        model = PipelineModel.load(model_dir)

        # Make predictions
        logger.info("Making predictions on the test dataset...")
        predictions = model.transform(test_df)

        # Evaluate predictions
        evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
        accuracy = evaluator.evaluate(predictions)
        f1_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
        f1_score = f1_evaluator.evaluate(predictions)

        logger.info(f"Test Dataset Accuracy: {accuracy}")
        logger.info(f"Test Dataset F1 Score: {f1_score}")

        # Stop Spark session
        spark.stop()

    except Exception as e:
        logger.error(f"An error occurred during testing: {e}")




