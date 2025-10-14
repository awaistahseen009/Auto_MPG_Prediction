import boto3
from botocore.client import Config
from dotenv import load_dotenv
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import make_logger
import os
import zipfile
load_dotenv()

ACCESS_KEY_ID = os.getenv("ACCESS_KEY_ID")
SECRET_ACCESS_KEY = os.getenv("SECRET_ACCESS_KEY")
BUCKET_NAME = os.getenv("BUCKET_NAME")
print(BUCKET_NAME)
FILE_KEY = os.getenv("FILE_KEY")
os.makedirs("data", exist_ok=True) 
LOCAL_FILE_PATH =  os.path.join("data",os.getenv("FILE_KEY",) )
ENDPOINT_URL = os.getenv("ENDPOINT_URL")
EXTRACT_DIR=os.getenv("EXTRACT_DIR")

logger = make_logger("data_ingestion", "DEBUG")
def download_file()->None:

    boto_client = boto3.client(
    's3',
    aws_access_key_id=ACCESS_KEY_ID,
    aws_secret_access_key=SECRET_ACCESS_KEY,
    config=Config(signature_version='s3v4'),
    endpoint_url=ENDPOINT_URL,
    region_name='sfo3'
    )
    try:
        logger.debug("Downloading %s from bucket %s", FILE_KEY, BUCKET_NAME )
        boto_client.download_file(BUCKET_NAME, FILE_KEY, LOCAL_FILE_PATH)
        logger.debug("Download completed and saved on location %s", LOCAL_FILE_PATH )
    except Exception as e:
        logger.error("Error during download: %s", e)

def extract_file(filepath:str)->None:
    try:
        logger.debug("Starting Extracting files from %s", LOCAL_FILE_PATH)
        os.makedirs(EXTRACT_DIR, exist_ok=True)
        with zipfile.ZipFile(LOCAL_FILE_PATH, "r") as zip_file:
            zip_file.extractall(path=EXTRACT_DIR)
        logger.info(f"Extraction complete! Files extracted to {EXTRACT_DIR}")
        os.remove(LOCAL_FILE_PATH)
    except zipfile.BadZipFile:
        logger.error(f"Error: {LOCAL_FILE_PATH} is not a valid ZIP file.")


if __name__=="__main__":
    download_file()
    extract_file(LOCAL_FILE_PATH)