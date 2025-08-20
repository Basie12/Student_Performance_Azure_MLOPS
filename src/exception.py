import sys
import logging 

def error_message_detail(error, error_detail):
    """
    Extracts detailed error information including file name, line number, and error message.
    
    Args:
        error: The original error or message.
        error_detail: The sys.exc_info() tuple.
    
    Returns:
        str: Formatted error message.
    """
    _, _, exc_tb = error_detail
    if exc_tb is None:
        return f"Error: {str(error)} (No traceback available)"
    
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = f"Error occurred in Python script [{file_name}] at line [{exc_tb.tb_lineno}] with message: {str(error)}"
    return error_message

class CustomException(Exception):
    """
    Custom exception class that captures and formats detailed error information.
    Optionally logs the error if logging is configured.
    """
    def __init__(self, error_message, error_detail):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail)
        
        # Optionally log the error (integrate with your logger)
        logging.error(self.error_message)

    def __str__(self):
        return self.error_message

    def __repr__(self):
        return f"CustomException({self.error_message})"



import os
import sys
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        self.model_path = os.path.join('artifact', 'model.pkl')
        self.preprocessor_path = os.path.join('artifact', 'preprocessor.pkl')

    def predict(self, features):
        try:
            # Validate input DataFrame
            if not isinstance(features, pd.DataFrame):
                raise CustomException("Input features must be a pandas DataFrame", sys)
            expected_columns = [
                'gender', 'race_ethnicity', 'parental_level_of_education',
                'lunch', 'test_preparation_course', 'reading_score', 'writing_score'
            ]
            if not all(col in features.columns for col in expected_columns):
                raise CustomException(f"Input missing required columns: {expected_columns}", sys)
            if list(features.columns) != expected_columns:
                logging.warning("Reordering columns to match training data")
                features = features[expected_columns]

            # Load model and preprocessor
            logging.info(f"Loading model from {self.model_path}")
            model = load_object(file_path=self.model_path)
            logging.info(f"Loading preprocessor from {self.preprocessor_path}")
            preprocessor = load_object(file_path=self.preprocessor_path)

            # Transform and predict
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds

        except Exception as e:
            logging.error(f"Error in prediction: {str(e)}")
            raise CustomException(e, sys)

class CustomData:
    def __init__(self,
                 gender: str,
                 ethnicity: str,
                 parental_level_of_education: str,
                 lunch: str,
                 test_preparation_course: str,
                 reading_score: str,
                 writing_score: str):
        self.gender = gender
        self.race_ethnicity = ethnicity  # Maps to pipeline's column name
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        try:
            self.reading_score = int(float(reading_score))
            self.writing_score = int(float(writing_score))
            if not (0 <= self.reading_score <= 100 and 0 <= self.writing_score <= 100):
                raise ValueError("Scores must be between 0 and 100")
        except (ValueError, TypeError) as e:
            raise CustomException(f"Invalid score input: {str(e)}", sys)
        # Define expected columns here
        self.expected_columns = [
            'gender', 'race_ethnicity', 'parental_level_of_education',
            'lunch', 'test_preparation_course', 'reading_score', 'writing_score'
        ]

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score]
            }
            df = pd.DataFrame(custom_data_input_dict)[self.expected_columns]
            logging.info("Created DataFrame from custom input")
            return df

        except Exception as e:
            logging.error(f"Error creating DataFrame: {str(e)}")
            raise CustomException(e, sys)


# Example usage (runs only if executed directly, for testing)
if __name__ == "__main__":
    try:
        # Simulate an error
        division_by_zero = 1 / 0
    except Exception as e:
        custom_exc = CustomException("Division by zero error occurred", sys.exc_info())
        raise custom_exc