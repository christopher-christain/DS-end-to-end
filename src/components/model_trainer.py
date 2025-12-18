import os
import sys
import joblib
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn

from dataclasses import dataclass
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV

from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor

from src.components.data_transformation import DataTransformation
from src.exception import CustomException
from src.logger import logging


# ---------------- CONFIG ---------------- #

@dataclass
class ModelTrainerConfig:
    best_model_path: str = os.path.join("artifacts", "best_model.pkl")
    metrics_path: str = os.path.join("artifacts", "model_metrics.csv")


# ---------------- TRAINER ---------------- #

class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()

        # MLflow setup
        mlflow.set_tracking_uri("file:./mlruns")
        mlflow.set_experiment("student_performance_experiment")

    def evaluate_model(self, y_true, y_pred):
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        return mae, rmse, r2

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Starting model training with MLflow tracking")

            X_train = train_array[:, :-1]
            y_train = train_array[:, -1]
            X_test = test_array[:, :-1]
            y_test = test_array[:, -1]

            models_with_params = {
                "Ridge": {
                    "model": Ridge(),
                    "params": {
                        "alpha": [0.1, 1.0, 10.0, 50.0, 100.0]
                    }
                },
                "KNN": {
                    "model": KNeighborsRegressor(),
                    "params": {
                        "n_neighbors": [3, 5, 7, 9, 11],
                        "weights": ["uniform", "distance"]
                    }
                },
                "RandomForest": {
                    "model": RandomForestRegressor(random_state=42),
                    "params": {
                        "n_estimators": [100, 200, 300],
                        "max_depth": [None, 10, 20, 30],
                        "min_samples_split": [2, 5, 10]
                    }
                },
                "AdaBoost": {
                    "model": AdaBoostRegressor(random_state=42),
                    "params": {
                        "n_estimators": [50, 100, 200],
                        "learning_rate": [0.01, 0.1, 1.0]
                    }
                }
            }

            results = []
            best_model = None
            best_score = -float("inf")
            best_model_name = None

            for model_name, model_data in models_with_params.items():

                logging.info(f"Tuning and training model: {model_name}")

                with mlflow.start_run(run_name=model_name):

                    mlflow.log_param("model_name", model_name)

                    search = RandomizedSearchCV(
                        estimator=model_data["model"],
                        param_distributions=model_data["params"],
                        n_iter=10,
                        cv=5,
                        scoring="r2",
                        n_jobs=-1,
                        random_state=42
                    )

                    search.fit(X_train, y_train)

                    best_estimator = search.best_estimator_
                    y_pred = best_estimator.predict(X_test)

                    mae, rmse, r2 = self.evaluate_model(y_test, y_pred)

                    # Log metrics
                    mlflow.log_metric("MAE", mae)
                    mlflow.log_metric("RMSE", rmse)
                    mlflow.log_metric("R2", r2)

                    # Log best hyperparameters
                    mlflow.log_params(search.best_params_)

                    # Log model artifact
                    mlflow.sklearn.log_model(
                        best_estimator,
                        name="model"
                    )

                    results.append({
                        "model": model_name,
                        "MAE": mae,
                        "RMSE": rmse,
                        "R2": r2
                    })

                    if r2 > best_score:
                        best_score = r2
                        best_model = best_estimator
                        best_model_name = model_name

            # Save metrics
            metrics_df = pd.DataFrame(results)
            os.makedirs(os.path.dirname(self.config.metrics_path), exist_ok=True)
            metrics_df.to_csv(self.config.metrics_path, index=False)

            # Save best model
            joblib.dump(best_model, self.config.best_model_path)

            logging.info(f"Best Model: {best_model_name} | R2 Score: {best_score}")

            return best_model_name, best_score

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    config_path = "src/config/config.yaml"

    transformer = DataTransformation(config_path)
    train_arr, test_arr = transformer.initiate_data_transformation()

    trainer = ModelTrainer()
    best_model, best_r2 = trainer.initiate_model_trainer(train_arr, test_arr)

    print(f"Best model: {best_model}, R2 score: {best_r2}")
