import os
import yaml
import pandas as pd
from typing import Dict
from src.exception import CustomException
import sys


def read_yaml(file_path: str) -> Dict:
    """
    Reads a YAML file and returns its contents as a dictionary.
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"YAML file not found at: {file_path}")

        with open(file_path, "r") as yaml_file:
            return yaml.safe_load(yaml_file)

    except Exception as e:
        raise CustomException(e, sys)


class CSVDataSource:
    """
    Data source class for loading CSV files from local disk.
    """

    def __init__(self, file_path: str):
        self.file_path = file_path

    def load_data(self) -> pd.DataFrame:
        try:
            if not os.path.exists(self.file_path):
                raise FileNotFoundError(f"CSV file not found at: {self.file_path}")

            return pd.read_csv(self.file_path)

        except Exception as e:
            raise CustomException(e, sys)


def load_data_source(config: Dict):
    """
    Factory function that returns the appropriate data source object.
    """
    try:
        source_type = config["data_source"]["type"].lower()

        if source_type == "csv":
            return CSVDataSource(config["data_source"]["path"])

        raise ValueError(f"Unsupported data source type: {source_type}")

    except Exception as e:
        raise CustomException(e, sys)
