import os
import sys
import pandas as pd
import joblib

from src.exception import CustomException
from src.logger import logging


class PredictPipeline:
    def __init__(self):
        try:
            self.model_path = os.path.join("artifacts", "best_model.pkl")
            self.preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

            self.model = joblib.load(self.model_path)
            self.preprocessor = joblib.load(self.preprocessor_path)

            logging.info("Model and preprocessor loaded successfully")

        except Exception as e:
            raise CustomException(e, sys)

    def predict(self, features: pd.DataFrame):
        try:
            logging.info("Starting prediction")

            transformed_data = self.preprocessor.transform(features)
            predictions = self.model.predict(transformed_data)

            return predictions

        except Exception as e:
            raise CustomException(e, sys)
