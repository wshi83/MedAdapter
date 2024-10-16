import os
import logging
from logging import FileHandler, Formatter
from datetime import datetime

# Initial log directory
BASE_DIR = f"logs/{datetime.now().strftime('%Y%m%d-%H%M')}"
DIR = BASE_DIR
os.makedirs(DIR, exist_ok=True)

def get_log_dir():
    return DIR

def get_base_dir():
    return BASE_DIR

def update_log_folder(new_dir, process_index):
    global DIR
    DIR = f"{BASE_DIR}/{new_dir}/gpu_{process_index}"
    os.makedirs(DIR, exist_ok=True)

    # Update file handlers for each logger
    for logger_name, logger in loggers.items():
        # Remove old file handlers
        for handler in logger.handlers[:]:
            if isinstance(handler, FileHandler):
                logger.removeHandler(handler)
                
        if logger_name in ["train", "eval"]:
            # For 'train' and 'eval' loggers, use the shared directory
            logger_dir = f"{BASE_DIR}/{new_dir}"
        else:
            # For other loggers, create a separate directory for each GPU process
            logger_dir = DIR
    
        # Create a new file handler with the updated directory
        file_name = logger_file_map[logger.name]
        new_file_handler = FileHandler(f"{logger_dir}/{file_name}")
        new_file_handler.setFormatter(Formatter('%(message)s'))
        logger.addHandler(new_file_handler)

# Store loggers and their corresponding file names
logger_file_map = {
    "api": "api_response.log",
    "search": "search_procedure.log",
    "train": "training_text.log",
    "tensor": "training_tensor.log",
    "eval": "eval_details.log",
    "error": "error.log",
    "adaptor": "adaptor_rating.log",
}

# Initialize loggers
loggers = {name: logging.getLogger(name) for name in logger_file_map}
for logger in loggers.values():
    logger.setLevel(logging.DEBUG)