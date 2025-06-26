import os
import logging
from datetime import datetime
from zoneinfo import ZoneInfo

def setup_logger(name=None, console_log_level=None, file_log_level=None):
    # Get log levels from environment variables
    console_log_level = (console_log_level or os.getenv("CLL", "INFO")).upper()
    file_log_level = (file_log_level or os.getenv("FLL", "")).upper()
    log_file_path = os.getenv("LOG_FILE_PATH", ".")  # Default to current directory if LOG_FILE_PATH is not set
    
    taipei_time = datetime.now(ZoneInfo("Asia/Taipei"))
    current_time = taipei_time.strftime("%Y%m%d_%H%M%S")
    log_file_path = os.path.join(log_file_path, f"app_{current_time}.log")
    
    # Create logger
    logger = logging.getLogger(name)
    
    # Prevent duplicated handler
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Set the global level for the logger from console_log_level (higher priority)
    logger.setLevel(logging.DEBUG)

    # Set logger format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Console logging
    if console_log_level:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(console_log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File logging
    if file_log_level:
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(file_log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
