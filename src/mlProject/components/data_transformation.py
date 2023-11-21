from src.mlProject.entity.config_entity import DataTransformationConfig
import os
from src.mlProject import logger
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OrdinalEncoder,OneHotEncoder
import numpy as np
from src.mlProject.constants import *
from src.mlProject.utils.common import *


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config



    def get_data_transformer_object(self):
        try:

            Numerical_col_1  = ['temperature','unit_price', 'timestamp_day_of_month', 'timestamp_day_of_week','timestamp_hour']

            Numerical_col_2  = ['quantity']

            categorical_col_for_OnehotEncoding = ['category']


            Numerical_pipeline_one_for_missingvalues = Pipeline(
                            steps=[
                            ('scaler',StandardScaler())
                            ])

            Numerical_pipeline_two_for_missingvalues = Pipeline(
                            steps=[
                            ('imputer',SimpleImputer(strategy='constant', fill_value=0)),
                            ('scaler',StandardScaler())
                            ])
                        
            categorical_pipeline_for_OnehotEncoding = Pipeline(
                                        steps=[
                                        ('one_hot_encoder', OneHotEncoder(sparse_output=False,handle_unknown = 'ignore')),
                                        ('scaler',StandardScaler())
                                        ]
                                        )

            preprocessor=ColumnTransformer(transformers= 
                    [('Numerical_pipeline_1_for_missingvalues', Numerical_pipeline_one_for_missingvalues,Numerical_col_1),
                    ('Numerical_pipeline_2_for_missingvalues', Numerical_pipeline_two_for_missingvalues,Numerical_col_2),
                    ('categorical_pipeline_for_OnehotEncoding', categorical_pipeline_for_OnehotEncoding,categorical_col_for_OnehotEncoding )
                    ],
                                                                    remainder='passthrough',sparse_threshold=0)     
            
            return preprocessor
            

        except Exception as e:
            raise e
    
    def initiate_data_transformation(self,):
        try:
            
            train_df = pd.read_csv(self.config.train_file_path)
            test_df = pd.read_csv(self.config.test_file_path)
            preprocessor = self.get_data_transformer_object()


            #training dataframe
            input_feature_train_df = train_df.drop(columns='estimated_stock_pct', axis=1)
            target_feature_train_df = train_df['estimated_stock_pct']

            #testing dataframe
            input_feature_test_df = test_df.drop(columns='estimated_stock_pct', axis=1)
            target_feature_test_df = test_df['estimated_stock_pct']

            preprocessor_object = preprocessor.fit(input_feature_train_df)
            input_feature_train_final = preprocessor_object.transform(input_feature_train_df)
            input_feature_test_final =preprocessor_object.transform(input_feature_test_df)


            train_arr = np.c_[input_feature_train_final, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_final, np.array(target_feature_test_df)]


            #save numpy array data

            save_numpy_array(train_arr, self.config.transformed_train_file_path, 'train_array.npy')
            save_numpy_array(test_arr, self.config.transformed_train_file_path, 'test_array.npy')
            save_preprocessor(preprocessor_object, self.config.transformed_object_file_path)

            
            return (
                train_arr,
                test_arr,
                self.config.transformed_object_file_path, preprocessor_object
            )
        
        except Exception as e:
            raise e
            

