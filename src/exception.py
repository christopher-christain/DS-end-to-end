import sys
import traceback
from src.logger import logging


def error_message_details(error: Exception, error_detail=sys) -> str:
    """
    Extracts detailed error information including file name, line number, and exception type.
    """
    exc_type, exc_value, exc_tb = error_detail.exc_info()

    if exc_tb is None:
        return f"{type(error).__name__}: {error}"

    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno

    return (
        f"{type(error).__name__} occurred in script: [{file_name}] "
        f"on line: [{line_number}] → {error}"
    )



class CustomException(Exception):
    """
    Custom exception that provides detailed traceback information
    and convenient logging / serialization methods.
    """
    def __init__(self, error: Exception, error_detail=sys):
        self.message = error_message_details(error, error_detail)
        super().__init__(self.message)

    def __str__(self) -> str:
        return self.message

    def __repr__(self) -> str:
        return f"CustomException({self.message})"
    
    def get_traceback(self) -> str:
        """
        Returns the full traceback as a string.
        """
        return "".join(traceback.format_exception(type(self), self, self.__traceback__))

    def log_exception(self, logger=None) -> None:
        """
        Logs the exception using the provided logger.
        If no logger is provided, logs to console.
        """
        if logger:
            logger.error(self.get_traceback())
            logger.error(self.message)
        else:
            print(self.get_traceback())
            print(self.message)

    def to_dict(self) -> dict:
        """
        Returns the exception details as a dictionary.
        Useful for structured logging or monitoring systems.
        """
        return {
            "type": type(self).__name__,
            "message": self.message,
            "traceback": self.get_traceback()
        }

