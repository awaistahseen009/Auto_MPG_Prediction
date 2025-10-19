import pytest as test
import os
from pathlib import Path
import zipfile
import pandas as pd
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_ingestion import extract_file , download_file
from config import column_names
import shutil


@test.fixture(scope="session")
def prepare_data_dir(tmp_path_factory):
    """Create a single temporary data dir for all tests."""
    data_dir = tmp_path_factory.mktemp("data")
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    return raw_dir
@test.mark.order(1)
def test_download_file_creates_zip(prepare_data_dir):
    """Ensure download_file creates a valid zip file with expected properties."""
    download_file(pathname=prepare_data_dir)  

    zip_files = list(prepare_data_dir.glob("*.zip"))
    assert len(zip_files) == 1, "No ZIP file found after download."
    
    zip_path = zip_files[0]

    assert zip_path.stat().st_size > 1000, "Downloaded file size too small."
    assert zip_path.suffix == ".zip", "File extension is not .zip."
    
    
    assert zipfile.is_zipfile(zip_path), "Downloaded file is not a valid ZIP."

@test.mark.order(2)
def test_extract_file_from_download(prepare_data_dir, tmp_path):
    """Test extracting the ZIP downloaded in previous step."""
    zip_files = list(prepare_data_dir.glob("*.zip"))
    assert len(zip_files) == 1, "No ZIP file found for extraction."

    zip_path = zip_files[0]
    extract_dir = tmp_path / "extracted"
    extract_dir.mkdir(parents=True, exist_ok=True)

    extract_file(filepath=str(zip_path),extracted_path=str(extract_dir))
    extracted_files = list(extract_dir.glob("*"))
    assert len(extracted_files) > 0, "No files extracted from ZIP."

    csv_files = [f for f in extracted_files if f.suffix == ".data"]
    assert len(csv_files) > 0, "No Data file found after extraction."

@test.mark.order(3)
def test_features_dataset(prepare_data_dir, tmp_path):
    extracted_dir = tmp_path / "extracted"
    zip_files = list(prepare_data_dir.glob("*.zip"))
    assert len(zip_files) == 1, "No ZIP file found for feature test."
    zip_path = zip_files[0]

    extract_file(filepath=str(zip_path), extracted_path=str(extracted_dir))

    extracted_files = list(extracted_dir.rglob("*.data"))
    assert extracted_files, f"No .data file found in {extracted_dir}"
    data_file = extracted_files[0]

    df = pd.read_csv(data_file, sep=r"\s+", header=None, names=column_names, na_values="?")
    assert df.shape[1] == 9, f"Expected 9 columns, got {df.shape[1]}"
    assert df.shape[0] == 398, f"Expected 398 rows, got {df.shape[0]}"

