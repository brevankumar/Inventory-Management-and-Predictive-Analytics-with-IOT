from flask import Flask, render_template, request
import os 
import numpy as np
import pandas as pd
from src.mlProject.pipeline.prediction import PredictionPipeline,CustomData


app = Flask(__name__) # initializing a flask app

@app.route('/',methods=['GET'])  # route to display the home page
def homePage():
    return render_template("index.html")


@app.route('/train',methods=['GET'])  # route to train the pipeline
def training():
    os.system("python main.py")
    return "Training Successful!" 


@app.route('/predict',methods=['GET','POST'])

def predict_datapoint():
    if request.method=='GET':
        return render_template('form.html')
    
    else:
        data=CustomData(
            
            #estimated_stock_pct = float(request.form.get('estimated_stock_pct')),
            category  = request.form.get('category'),
            temperature = float(request.form.get('temperature')),
            quantity = float(request.form.get('quantity')),
            unit_price = float(request.form.get('unit_price')),
            timestamp_day_of_month = int(request.form.get('timestamp_day_of_month')),
            timestamp_day_of_week = int(request.form.get('timestamp_day_of_week')),
            timestamp_hour = int(request.form.get('timestamp_hour')),
        )

        final_new_data=data.get_data_as_dataframe()
        predict_pipeline=PredictionPipeline()
        pred=predict_pipeline.predict(final_new_data)

        result = pred[0]
        
        return render_template('results.html',final_result=result)




if __name__=="__main__":
    app.run(host='0.0.0.0',debug=True,port=5000)