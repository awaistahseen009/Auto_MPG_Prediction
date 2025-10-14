import logging
import os
from dotenv import load_dotenv
load_dotenv()
DIR_NAME = os.getenv("LOG_DIR")
os.makedirs(DIR_NAME, exist_ok=True)

def make_logger(name:str, level:str):
    logger  = logging.getLogger(name)
    logger.setLevel(level)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)

    filename = os.path.join(DIR_NAME, f"{name}.log")

    file_handler = logging.FileHandler(filename=filename)
    file_handler.setLevel(level)

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger
