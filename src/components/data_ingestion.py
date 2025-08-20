import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
# from src.components.data_transformation import DataTransformationConfig 
from src.components.model_trainer import ModelTrainerConfig, ModelTrainer  

import multiprocessing
# multiprocessing.set_start_method('spawn')
# multiprocessing.set_start_method('forkserver')

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifact', 'train.csv')
    test_data_path: str = os.path.join('artifact', 'test.csv')
    raw_data_path: str = os.path.join('artifact', 'data.csv')
    dataset_path: str = os.path.join('notebook', 'data', 'StudentsPerformance.csv')  # Configurable path

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method")
        try:
            # Check if dataset exists
            if not os.path.exists(self.ingestion_config.dataset_path):
                raise CustomException(f"Dataset file not found at {self.ingestion_config.dataset_path}", sys)
            
            df = pd.read_csv(self.ingestion_config.dataset_path)
            if df.empty:
                raise CustomException("Dataset is empty", sys)
            logging.info(f'Read the dataset as a dataframe from {self.ingestion_config.dataset_path}')
            
            # Clean column names: replace spaces and slashes with underscores, and convert to lowercase
            df.columns = df.columns.str.replace(" ", "_").str.replace("/", "_").str.lower()
            
            # Create artifact directory once
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            
            df.to_csv(self.ingestion_config.raw_data_path, index=False)
            logging.info("Train test split initiated")
            
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path, index=False)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False)
            logging.info("Ingestion of data is completed")

            return (self.ingestion_config.train_data_path, self.ingestion_config.test_data_path)

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    obj = DataIngestion()
    try:
        train_data, test_data = obj.initiate_data_ingestion()
        
        data_transformation = DataTransformation()
        train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transformation(train_data, test_data)
        logging.info(f"Data transformation completed. Preprocessor saved at {preprocessor_path}")

        model_trainer = ModelTrainer()
        r2_score = model_trainer.initiate_model_training(train_arr, test_arr)
        logging.info(f"Model training completed. Best model R² score: {r2_score}")
        print(f"Best model R² score: {r2_score}")
        
    except Exception as e:
        logging.error(f"Error in pipeline: {str(e)}")
        raise CustomException(e, sys)