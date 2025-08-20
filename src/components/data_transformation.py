import os
import sys
from dataclasses import dataclass, field  # Added 'field' import
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifact', 'preprocessor.pkl')
    numerical_features: list = field(default_factory=lambda: ["reading_score", "writing_score"])
    categorical_features: list = field(default_factory=lambda: ["gender", "race_ethnicity", "parental_level_of_education", "lunch", "test_preparation_course"])
    target_column_name: str = "math_score"

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        Creates and returns a preprocessor object for numerical and categorical features.

        Returns:
            ColumnTransformer: Preprocessor object for transforming features.
        Raises:
            CustomException: If an error occurs during pipeline creation.
        """
        try:
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder())
                    # Removed StandardScaler for categorical features
                ]
            )

            logging.info(f"Numerical columns: {self.data_transformation_config.numerical_features}")
            logging.info(f"Categorical columns: {self.data_transformation_config.categorical_features}")

            preprocessor = ColumnTransformer([
                ("num_pipeline", num_pipeline, self.data_transformation_config.numerical_features),
                ("cat_pipeline", cat_pipeline, self.data_transformation_config.categorical_features)
            ])

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_data_path: str, test_data_path: str):
        """
        Applies preprocessing to train and test data and saves the preprocessor object.

        Args:
            train_data_path (str): Path to the training data CSV.
            test_data_path (str): Path to the test data CSV.

        Returns:
            tuple: (train_array, test_array, preprocessor_file_path)
        Raises:
            CustomException: If an error occurs during data transformation.
        """
        try:
            # Validate file paths
            if not os.path.exists(train_data_path) or not os.path.exists(test_data_path):
                raise FileNotFoundError("Train or test data file not found")

            train_df = pd.read_csv(train_data_path)
            test_df = pd.read_csv(test_data_path)
            logging.info("Read train and test data")

            # Validate required columns
            required_columns = self.data_transformation_config.numerical_features + \
                               self.data_transformation_config.categorical_features + \
                               [self.data_transformation_config.target_column_name]
            if not all(col in train_df.columns for col in required_columns):
                raise ValueError("Required columns missing in training data")
            if not all(col in test_df.columns for col in required_columns):
                raise ValueError("Required columns missing in test data")

            logging.info("Obtaining preprocessing object")
            preprocessor_obj = self.get_data_transformer_object()

            # Split features and target
            input_feature_train_df = train_df.drop(columns=[self.data_transformation_config.target_column_name], axis=1)
            target_feature_train_df = train_df[self.data_transformation_config.target_column_name]
            input_feature_test_df = test_df.drop(columns=[self.data_transformation_config.target_column_name], axis=1)
            target_feature_test_df = test_df[self.data_transformation_config.target_column_name]

            # Validate target column for missing values
            if target_feature_train_df.isnull().any() or target_feature_test_df.isnull().any():
                raise ValueError("Target column contains missing values")

            logging.info("Applying preprocessing object on training and testing dataframes")
            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)

            # Combine features and target
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object to {self.data_transformation_config.preprocessor_obj_file_path}")
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    # Example usage
    data_transformation = DataTransformation()
    train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transformation(
        train_data_path="artifact/train.csv",
        test_data_path="artifact/test.csv"
    )