import logging
import os
from datetime import datetime

def setup_logger(logs_dir_name="logs"):
    """
    Sets up a timestamped logger in a centralized directory.
    Returns:
        str: Path to the created log file.
    """
    # Define the log file name with full timestamp
    log_file = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
    
    # Define the logs directory path (relative to project root or current dir)
    logs_dir = os.path.join(os.getcwd(), logs_dir_name)
    
    # Create the logs directory if it doesn't exist
    try:
        os.makedirs(logs_dir, exist_ok=True)
    except OSError as e:
        raise RuntimeError(f"Failed to create logs directory: {e}")
    
    # Define the full log file path
    log_file_path = os.path.join(logs_dir, log_file)
    
    # Configure logging
    logging.basicConfig(
        filename=log_file_path,
        format="[%(asctime)s, %(lineno)d, %(levelname)s, %(message)s]",
        level=logging.INFO
    )
    
    return log_file_path

# Setup the logger (call this in your main script or on import if needed)
LOG_FILE_PATH = setup_logger()