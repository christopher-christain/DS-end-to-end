import os
from dataclasses import dataclass
from src.utils import read_yaml
from src.exception import CustomException
import sys


@dataclass
class DataIngestionConfig:
    raw_data_path: str
    train_data_path: str
    test_data_path: str
    test_size: float
    data_source_config_path: str


def get_data_ingestion_config(config_path: str) -> DataIngestionConfig:
    """
    Reads config.yaml and returns a DataIngestionConfig object
    """
    try:
        config = read_yaml(config_path)
        ingestion_config = config["data_ingestion"]

        return DataIngestionConfig(
            raw_data_path=ingestion_config["raw_data_path"],
            train_data_path=ingestion_config["train_data_path"],
            test_data_path=ingestion_config["test_data_path"],
            test_size=ingestion_config["test_size"],
            data_source_config_path=ingestion_config["data_source_config_path"],
        )

    except Exception as e:
        raise CustomException(e, sys)
