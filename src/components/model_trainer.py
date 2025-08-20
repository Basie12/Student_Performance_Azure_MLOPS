import os
import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, AdaBoostRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR  # Imported but not used; can remove if not needed
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import RandomizedSearchCV
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
import warnings

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifact", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self, train_array, test_array):
        try:
            logging.info("Splitting training and testing data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "XGBoost Regressor": XGBRegressor(),
                "CatBoost Regressor": CatBoostRegressor(verbose=False),  # Set verbose=False to suppress output
                "AdaBoost Regressor": AdaBoostRegressor()
            }

            # Inline model evaluation (removed separate evaluate_models function)
            model_report = {}
            for model_name, model in models.items():
                try:
                    logging.info(f"Training {model_name}")
                    model.fit(X_train, y_train)
                    y_test_pred = model.predict(X_test)
                    test_model_score = r2_score(y_test, y_test_pred)
                    model_report[model_name] = test_model_score
                    logging.info(f"{model_name} R² score: {test_model_score}")
                except Exception as e:
                    logging.warning(f"Error training/evaluating {model_name}: {str(e)}")
                    model_report[model_name] = -1  # Mark as failed

            # Get the best model score
            best_model_score = max(sorted(model_report.values()))
            # Get best model name
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                logging.warning(f"Best model score {best_model_score} is below 0.6 threshold - model not saved")
                raise CustomException("No best model found with sufficient score", sys)
            
            # Print/log model and R² before saving
            logging.info(f"Best model before saving: {best_model}")
            logging.info(f"Best model R² before saving: {best_model_score}")
            print(f"Best model: {best_model}")
            print(f"Best model R²: {best_model_score}")

            logging.info(f"Best model found on both training and testing dataset: {best_model_name} with R²: {best_model_score}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)

            return r2_square

        except Exception as e:
            raise CustomException(e, sys)