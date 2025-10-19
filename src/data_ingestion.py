import boto3
from botocore.client import Config
from dotenv import load_dotenv
import sys
import os
import zipfile

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import make_logger

load_dotenv()

ACCESS_KEY_ID = os.getenv("ACCESS_KEY_ID")
SECRET_ACCESS_KEY = os.getenv("SECRET_ACCESS_KEY")
BUCKET_NAME = os.getenv("BUCKET_NAME")
FILE_KEY = os.getenv("FILE_KEY")
ENDPOINT_URL = os.getenv("ENDPOINT_URL")

RAW_DATA_DIR = "data/raw"
os.makedirs(RAW_DATA_DIR, exist_ok=True)

LOCAL_FILE_PATH = os.path.join(RAW_DATA_DIR, FILE_KEY)
EXTRACT_DIR = os.path.join(RAW_DATA_DIR, "extracted")

logger = make_logger("data_ingestion", "DEBUG")

def download_file(pathname=None) -> None:
    boto_client = boto3.client(
        's3',
        aws_access_key_id=ACCESS_KEY_ID,
        aws_secret_access_key=SECRET_ACCESS_KEY,
        config=Config(signature_version='s3v4'),
        endpoint_url=ENDPOINT_URL,
        region_name='sfo3'
    )
    try:
        logger.debug("Downloading %s from bucket %s", FILE_KEY, BUCKET_NAME)
        if pathname:
            local_path = os.path.join(pathname, FILE_KEY)
            boto_client.download_file(BUCKET_NAME, FILE_KEY, local_path)
        else:
            boto_client.download_file(BUCKET_NAME, FILE_KEY, LOCAL_FILE_PATH)
        logger.debug("Download completed and saved at %s", LOCAL_FILE_PATH)
    except Exception as e:
        logger.error("Error during download: %s", e)
        raise

def extract_file(filepath: str, extracted_path = None) -> None:
    try:
        logger.debug("Starting extraction from %s", filepath)
        if extracted_path is not None:
            EXTRACT_DIR = extracted_path 
        else:
            EXTRACT_DIR = os.path.join(RAW_DATA_DIR, "extracted")
        os.makedirs(EXTRACT_DIR, exist_ok=True)
        with zipfile.ZipFile(filepath, "r") as zip_file:
            zip_file.extractall(path=EXTRACT_DIR)
        logger.debug("Extraction complete. Files are in %s", EXTRACT_DIR)
        # os.remove(filepath)
        logger.debug("Removed the original zip file %s", filepath)
    except zipfile.BadZipFile:
        logger.error("%s is not a valid ZIP file.", filepath)
        raise
    except Exception as e:
        logger.error("Error during extraction: %s", e)
        raise

if __name__ == "__main__":
    download_file()
    extract_file(LOCAL_FILE_PATH)
