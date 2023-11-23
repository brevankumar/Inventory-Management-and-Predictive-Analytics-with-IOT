import pandas as pd
import os
from src.mlProject import logger
import joblib
from src.mlProject.entity.config_entity import ModelTrainerConfig
from src.mlProject.constants import *
from src.mlProject.utils.common import *
from sklearn.ensemble import RandomForestRegressor



class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train_model(self,x_train,y_train):
        try:
            rf_reg = RandomForestRegressor(n_estimators = self.config.n_estimators,
                                            min_weight_fraction_leaf = self.config.min_weight_fraction_leaf,
                                            min_samples_split = self.config.min_samples_split,
                                            min_samples_leaf = self.config.min_samples_leaf,
                                            min_impurity_decrease = self.config.min_impurity_decrease,
                                            max_leaf_nodes = self.config.max_leaf_nodes,
                                            max_depth = self.config.max_depth)
                                                    
            rf_reg.fit(x_train,y_train)
            return rf_reg
        except Exception as e:
            raise e

    
    def train(self):
        train_arr = load_numpy_array_data(self.config.train_data_path)
        test_arr = load_numpy_array_data(self.config.test_data_path)


        x_train, y_train, x_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1],
            )
        
        model = self.train_model(x_train, y_train)


        joblib.dump(model, os.path.join(self.config.root_dir, self.config.model_name))
