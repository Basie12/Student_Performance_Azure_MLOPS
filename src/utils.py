import os
import sys
import dill
import logging
from src.exception import CustomException

def save_object(file_path: str, obj: object) -> None:
    """
    Save a Python object to a file using dill serialization.

    Args:
        file_path (str): Path to the file where the object will be saved.
        obj (object): The Python object to serialize and save.

    Raises:
        CustomException: If an error occurs during file operations or serialization.
    """
    try:
        if not file_path or not isinstance(file_path, str):
            raise ValueError("file_path must be a non-empty string")
        
        # Create directory if it doesn't exist
        dir_path = os.path.dirname(file_path)
        if dir_path:  # Only create directory if file_path includes a directory
            os.makedirs(dir_path, exist_ok=True)
        
        # Save the object using dill
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
        
        logging.info(f"Object successfully saved to {file_path}")

    except Exception as e:
        logging.error(f"Error saving object to {file_path}: {str(e)}")
        raise CustomException(e, sys)