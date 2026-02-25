import sys
import os
from src.logger import logger


def error_message_details(error: Exception, error_detail: sys):
    _, _, exc_tb = error_detail.exc_info()

    if exc_tb is not None:
        file_name = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno
    else:
        file_name = "Unknown"
        line_number = "Unknown"

    return f"Error occurred in script [{file_name}] at line [{line_number}] : {str(error)}"


class CustomException(Exception):
    def __init__(self, error_message: Exception, error_detail: sys):
        super().__init__(str(error_message))
        self.error_message = error_message_details(error_message, error_detail)

        # Log automatically when exception is created
        logger.error(self.error_message)

    def __str__(self):
        return self.error_message
