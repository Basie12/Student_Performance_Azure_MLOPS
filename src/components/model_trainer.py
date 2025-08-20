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

            # Define hyperparameter distributions for RandomizedSearchCV
            param_distributions = {
                "Random Forest": {
                    'n_estimators': [100, 200, 300],
                    'max_features': ['sqrt', 'log2', None],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                },
                "Decision Tree": {
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2', None]
                },
                "Gradient Boosting": {
                    'n_estimators': [100, 200, 300],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'max_depth': [3, 5, 7],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'subsample': [0.6, 0.8, 1.0]
                },
                "Linear Regression": {},  # No hyperparameters to tune
                "K-Neighbors Regressor": {
                    'n_neighbors': [3, 5, 7, 9],
                    'weights': ['uniform', 'distance'],
                    'p': [1, 2]
                },
                "XGBoost Regressor": {
                    'n_estimators': [100, 200, 300],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.6, 0.8, 1.0],
                    'colsample_bytree': [0.6, 0.8, 1.0]
                },
                "CatBoost Regressor": {
                    'iterations': [100, 200, 300],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'depth': [3, 5, 7],
                    'subsample': [0.6, 0.8, 1.0]
                },
                "AdaBoost Regressor": {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.05, 0.1, 0.5, 1.0],
                    'loss': ['linear', 'square', 'exponential']
                }
            }

            # model evaluation with hyperparameter tuning (reduced computation)
            model_report = {}
            best_models = {}  # To store the tuned models
            for model_name, model in models.items():
                try:
                    logging.info(f"Training and tuning {model_name}")
                    params = param_distributions.get(model_name, {})
                    if params:  # Only tune if params are defined
                        rs_cv = RandomizedSearchCV(
                            model, params, n_iter=5, cv=3, verbose=0, n_jobs=1, random_state=42  # Reduced n_iter=5, cv=3, n_jobs=1 for less computation
                        )
                        rs_cv.fit(X_train, y_train)
                        best_model_instance = rs_cv.best_estimator_
                        logging.info(f"Best params for {model_name}: {rs_cv.best_params_}")
                    else:
                        best_model_instance = model
                        best_model_instance.fit(X_train, y_train)
                    
                    y_test_pred = best_model_instance.predict(X_test)
                    test_model_score = r2_score(y_test, y_test_pred)
                    model_report[model_name] = test_model_score
                    best_models[model_name] = best_model_instance
                    logging.info(f"{model_name} tuned R² score: {test_model_score}")
                except Exception as e:
                    logging.warning(f"Error training/tuning {model_name}: {str(e)}")
                    model_report[model_name] = -1  # Mark as failed

            # Get the best model score
            best_model_score = max(sorted(model_report.values()))
            # Get best model name
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = best_models[best_model_name]

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