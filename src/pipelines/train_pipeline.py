import os
import sys
from src.logger import logging
from src.exception import CustomException
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

class TrainingPipeline:
    def __init__(self):
        self.ingestion = DataIngestion()
        self.transformation = DataTransformation()
        self.trainer = ModelTrainer()

    def run_pipeline(self):
        try:
            logging.info("Starting training pipeline")

            # Step 1: Data Ingestion
            logging.info("Running data ingestion")
            train_data_path, test_data_path = self.ingestion.initiate_data_ingestion()

            # Step 2: Data Transformation
            logging.info("Running data transformation")
            train_arr, test_arr, preprocessor_path = self.transformation.initiate_data_transformation(
                train_data_path, test_data_path
            )

            # Step 3: Model Training
            logging.info("Running model training")
            r2_score = self.trainer.initiate_model_training(train_arr, test_arr)

            logging.info(f"Training pipeline completed. Best model R² score: {r2_score}")
            print(f"Best model R² score: {r2_score}")

            return r2_score

        except Exception as e:
            logging.error(f"Error in training pipeline: {str(e)}")
            raise CustomException(e, sys)

if __name__ == "__main__":
    pipeline = TrainingPipeline()
    pipeline.run_pipeline()