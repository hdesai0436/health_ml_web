
from flask import Flask,jsonify,redirect, render_template,request
import json
import numpy as np
import pandas as pd
from file_operation.file_methods import file_operation
from predictfrommodel import predication


pred_model = predication()

app = Flask(__name__)

@app.route('/api',methods=['POST'])
def stroke_pred():
    content_type = request.headers.get('Content-Type')
    if (content_type == 'application/json'):
        d = request.get_json()
        gender = d['gender']
        age = d['age']
        hypertension = d['hypertension']
        heart_disease = d['heart_disease']
        ever_married = d['ever_married']
        work_type=d['work_type']
        Residence_type=d['Residence_type']
        avg_glucose_level=d['avg_glucose_level']
        bmi=d['bmi']
        smoking_status = d['smoking_status']
        test_data= np.array(list(d.values())).reshape(1,-1)
        pred = pred_model.predict(test_data)

        if pred == 1:
            return jsonify('stroke'),200
        else:
            return jsonify('no stroke'),200
      
       
    else:
        return jsonify('something went wrong'),404
    
    

if __name__=="__main__":
    app.run(debug=True)