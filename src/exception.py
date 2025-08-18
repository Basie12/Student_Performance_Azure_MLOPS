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

# Example usage (runs only if executed directly, for testing)
if __name__ == "__main__":
    try:
        # Simulate an error
        division_by_zero = 1 / 0
    except Exception as e:
        custom_exc = CustomException("Division by zero error occurred", sys.exc_info())
        raise custom_exc