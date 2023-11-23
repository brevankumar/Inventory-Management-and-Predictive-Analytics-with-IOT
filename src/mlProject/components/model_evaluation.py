import os
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from src.mlProject.utils.common import *
from src.mlProject.constants import *
from urllib.parse import urlparse
import numpy as np
import joblib
from src.mlProject.entity.config_entity import ModelEvaluationConfig
from pathlib import Path


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    
    def eval_metrics(self,actual, pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        return rmse, mae
    


    def save_results(self):

        train_arr = load_numpy_array_data(self.config.train_data_path)
        test_arr = load_numpy_array_data(self.config.test_data_path)

        x_train, y_train, x_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1],
            )


        model = joblib.load(self.config.model_path)

        
        predicted_qualities = model.predict(x_test)

        (rmse, mae) = self.eval_metrics(y_test, predicted_qualities)
        
        # Saving metrics as local
        scores = {"rmse": rmse, "mae": mae}
        save_json(path=Path(self.config.metric_file_name), data=scores)

