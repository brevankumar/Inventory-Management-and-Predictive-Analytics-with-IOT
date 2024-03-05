import joblib 
import numpy as np
import pandas as pd
from pathlib import Path


class PredictionPipeline:
    def __init__(self):
        self.model = joblib.load(Path('artifacts/model_trainer/model.joblib'))
        self.preprocessor = joblib.load(Path('artifacts/data_transformation/preprocessor_model.joblib'))

    
    def predict(self, data):
        
        preprocessor = self.preprocessor
        
        data_scaled = preprocessor.transform(data)
        
        prediction = self.model.predict(data_scaled)

        return prediction
    
        

class CustomData:
    def __init__( self, quantity: float, temperature: float, category: str, unit_price: float, timestamp_day_of_month: int,      
                 timestamp_day_of_week: int, timestamp_hour: int):
        
       # self.estimated_stock_pct=estimated_stock_pct
        self.quantity= quantity
        self.temperature = temperature      
        self.category = category   
        self.unit_price = unit_price 
        self.timestamp_day_of_month  = timestamp_day_of_month    
        self.timestamp_day_of_week  = timestamp_day_of_week  
        self.timestamp_hour =  timestamp_hour         
        

    def get_data_as_dataframe(self):
            custom_data_input_dict = {
               # 'estimated_stock_pct':[self.estimated_stock_pct],
                'quantity':[self.quantity],
                'temperature':[self.temperature],
                'category':[self.category],
                'unit_price':[self.unit_price],
                'timestamp_day_of_month':[self.timestamp_day_of_month],
                'timestamp_day_of_week':[self.timestamp_day_of_week],
                'timestamp_hour':[self.timestamp_hour]
            }
            
            return pd.DataFrame(custom_data_input_dict)
        
