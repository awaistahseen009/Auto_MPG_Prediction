import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import make_logger, column_names
from typing import List
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
logger = make_logger("data_preprocessing", "DEBUG")

def preprocess_data(path:str):
    scaler = StandardScaler()
    try:
        df = pd.read_csv(path, sep = r"\s+", header = None, names = column_names, na_values = "?")
        logger.debug("Data Loaded Successfully")
        # Removing the null values
        df.dropna(inplace=True)
        logger.debug("Null Values are removed")
        # droping the duplicates
        df.drop_duplicates(inplace=True)
        logger.debug("Duplicates are removed")
        numeric_columns = ["displacement", "horsepower", "weight", "acceleration"]
        df.drop("car_name", axis =1 , inplace = True)
        df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
        X = df.drop(["mpg"], axis = 1)
        y = df['mpg']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2 ,random_state = 42)
        X_train = pd.DataFrame(X_train, columns=X.columns)
        X_test = pd.DataFrame(X_test, columns=X.columns)
        y_train = pd.DataFrame(y_train, columns=['mpg'])
        y_test = pd.DataFrame(y_test, columns=['mpg'])
        train_df = pd.concat([X_train, y_train], axis=1)
        test_df = pd.concat([X_test, y_test], axis=1)
        train_df.to_csv("data/train_data.csv", index=False)
        logger.debug("Training dataset saved successfully")
        test_df.to_csv("data/test_data.csv", index=False)
        logger.debug("Testing dataset saved successfully")

    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logger.error('Failed to complete the data prprocessing process: %s', e)
        raise

if __name__=="__main__":
    preprocess_data("data/auto-mpg.data")