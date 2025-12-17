"""
1. Typical responsibilities inside data_ingestion.py
2.  Define data sources

Where the data comes from:

CSV / Excel files

Databases (PostgreSQL, MySQL)

APIs

Cloud storage (S3, GCS, Azure Blob)"""

import os
import sys
import pandas as pd
from src.logger import logging
from src.exception import CustomException
from src.components.data_validation import DataValidation
from src.utils import read_yaml, load_data_source
from src.config import get_data_ingestion_config
from sklearn.model_selection import train_test_split

class DataIngestion:
    def __init__(self, config_path: str):
        self.config = get_data_ingestion_config(config_path)
        self.data_validation = DataValidation(config_path)
        self.data_source_config = read_yaml(self.config.data_source_config_path)
        self.data_source = load_data_source(self.data_source_config)
    
    def initiate_data_ingestion(self) -> pd.DataFrame:
        try:
            logging.info("starting data ingestion process")

            # loading data from source
            df= self.data_source.load_data()
            logging.info("data loaded sucessfully from source")

            # validating the data
            self.data_validation.validate_data(df)
            logging.info("data validation completed")
    
    # saving raw data
            raw_data_path = self.config.raw_data_path
            os.makedirs(os.path.dirname(raw_data_path), exist_ok=True)
            df.to_csv(raw_data_path, index= False)
            logging.info(f"raw data saved at: {raw_data_path}")
            return df
        except Exception as e:
            raise CustomException(e, sys)
        
    def split_data(self, df: pd.DataFrame) -> tuple:
        try:
            logging.info("splitting data into train and test sets")
            train_data_path = self.config.train_data_path
            test_data_path = self.config.test_data_path

            os.makedirs(os.path.dirname(train_data_path), exist_ok=True)
            os.makedirs(os.path.dirname(test_data_path), exist_ok=True)

            train_set, test_set = train_test_split(df, test_size= self.config.test_size, random_state=42)
            train_set.to_csv(train_data_path, index= False)
            test_set.to_csv(test_data_path, index= False)
            logging.info(f"train and test data saved at: {train_data_path} and {test_data_path}")
            return train_set, test_set
        except Exception as e:
            raise CustomException(e, sys)
        
if __name__ == "__main__":
    config_path = "src/config/config.yaml"
    data_ingestion = DataIngestion(config_path)
    df = data_ingestion.initiate_data_ingestion()
    train_set, test_set = data_ingestion.split_data(df)