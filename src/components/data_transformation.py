"""
the data transormation.py file is responsible for transforming raw data into a format suitable for analysis or modeling.
this may include tasks such as cleaning the data, handling missing value, encoding categorical variables, feture scaling, amd feature engineering
 typical responsibilities inside data_transformatio.py
    1. Data Cleaning
    2. Feature Engineering
    3. encoding categorical variables
    4. Feature Scaling
    5. Dimensionality Reduction
    6. Data Transformation Pipelines

"""
import os
import sys
import pandas as pd
import numpy as np

from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import joblib

from src.exception import CustomException
from src.logger import logging
from src.config import get_data_ingestion_config

@dataclass
class DataTransformationConfig:
    preprocessor_path: str = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    def __init__(self, config_path: str):
        self.ingestion_config = get_data_ingestion_config(config_path)
        self.transformation_config = DataTransformationConfig()

    def get_preprocessor(self, df: pd.DataFrame):
        try:
            logging.info("Creating preprocessing pipelines")

            target_column = "math score"  # change if needed

            X = df.drop(columns=[target_column])

            categorical_features = X.select_dtypes(include="object").columns
            numerical_features = X.select_dtypes(exclude="object").columns

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(handle_unknown="ignore"))
                ]
            )

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", num_pipeline, numerical_features),
                    ("cat", cat_pipeline, categorical_features)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self):
        try:
            logging.info("Starting data transformation")

            train_path = self.ingestion_config.train_data_path
            test_path = self.ingestion_config.test_data_path

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            target_column = "math score"

            X_train = train_df.drop(columns=[target_column])
            y_train = train_df[target_column]

            X_test = test_df.drop(columns=[target_column])
            y_test = test_df[target_column]

            preprocessor = self.get_preprocessor(train_df)

            logging.info("Fitting preprocessor on training data")
            X_train_transformed = preprocessor.fit_transform(X_train)
            X_test_transformed = preprocessor.transform(X_test)

            train_array = np.c_[X_train_transformed, y_train.to_numpy()]
            test_array = np.c_[X_test_transformed, y_test.to_numpy()]

            os.makedirs(os.path.dirname(self.transformation_config.preprocessor_path), exist_ok=True)
            joblib.dump(preprocessor, self.transformation_config.preprocessor_path)

            logging.info("Preprocessor saved successfully")

            return train_array, test_array

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    config_path = "src/config/config.yaml"
    transformer = DataTransformation(config_path)
    train_arr, test_arr = transformer.initiate_data_transformation()
    print("Data transformation completed successfully")