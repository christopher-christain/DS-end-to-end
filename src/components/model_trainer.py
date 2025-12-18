import os
import sys
import numpy as np
import pandas as pd
import joblib

from dataclasses import dataclass
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor

from src.exception import CustomException
from src.logger import logging
from src.components.data_transformation import DataTransformation


@dataclass
class ModelTrainerConfig:
    best_model_path: str = os.path.join("artifacts", "best_model.pkl")
    metrics_path: str = os.path.join("artifacts", "model_metrics.csv")


class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()

    def evaluate_model(self, y_true, y_pred):
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        return mae, rmse, r2

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Starting model training")

            X_train = train_array[:, :-1]
            y_train = train_array[:, -1]

            X_test = test_array[:, :-1]
            y_test = test_array[:, -1]

            models = {
                "LinearRegression": LinearRegression(),
                "Ridge": Ridge(),
                "Lasso": Lasso(),
                "KNN": KNeighborsRegressor(),
                "DecisionTree": DecisionTreeRegressor(random_state=42),
                "RandomForest": RandomForestRegressor(random_state=42),
                "AdaBoost": AdaBoostRegressor(random_state=42),
            }

            results = []
            best_r2 = -float("inf")
            best_model = None
            best_model_name = None

            for model_name, model in models.items():
                logging.info(f"Training model: {model_name}")

                model.fit(X_train, y_train)

                y_pred = model.predict(X_test)

                mae, rmse, r2 = self.evaluate_model(y_test, y_pred)

                results.append({
                    "model": model_name,
                    "MAE": mae,
                    "RMSE": rmse,
                    "R2": r2
                })

                if r2 > best_r2:
                    best_r2 = r2
                    best_model = model
                    best_model_name = model_name

            metrics_df = pd.DataFrame(results)
            os.makedirs(os.path.dirname(self.config.metrics_path), exist_ok=True)
            metrics_df.to_csv(self.config.metrics_path, index=False)

            joblib.dump(best_model, self.config.best_model_path)

            logging.info(f"Best model: {best_model_name} with R2: {best_r2}")
            logging.info("Model training completed successfully")

            return best_model_name, best_r2

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    config_path = "src/config/config.yaml"

    transformer = DataTransformation(config_path)
    train_arr, test_arr = transformer.initiate_data_transformation()

    trainer = ModelTrainer()
    best_model, best_r2 = trainer.initiate_model_trainer(train_arr, test_arr)

    print(f"Best model: {best_model}, R2 score: {best_r2}")
