""" 2️⃣ Typical responsibilities inside data_ingestion.py
✅ 1. Define data sources

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
        

        