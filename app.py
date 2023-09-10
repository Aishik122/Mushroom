from flask import Flask,request,render_template
import numpy as np
import pandas as pd 

from src.pipeline.predict_pipeline import CustomData,PredictPipeline
from sklearn.preprocessing import StandardScaler



application = Flask(__name__)

app=application

## route for home page

@app.route('/')
def index():
    return render_template('home.html')
@app.route('/predictions',methods=['GET','POST'])
def predict_datapoints():
    if request.method == 'GET':
        return render_template('predictions.html')
    else:
        data =CustomData(
            odor=request.form.get('odor'),
            gill_size=request.form.get('gill_size'),
            gill_color=request.form.get('gill_color'),
            stalk_shape=request.form.get('stalk_shape'),
            stalk_root=request.form.get('stalk_root'),
            spore_print_color=request.form.get('spore_print_color'),
            population=request.form.get('population')
            
        )
        pred_df=data.get_data_frame()
        print(pred_df)
        
        predict_pipeline=PredictPipeline()
        
        results=predict_pipeline.predict(pred_df)
        return render_template('predictions.html',results)
    
if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True)