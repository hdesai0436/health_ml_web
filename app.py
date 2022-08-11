
from flask import Flask,jsonify,redirect, render_template,request
import json
import numpy as np
import pandas as pd
from predictfrommodel import predication
from application_logs.loger import app_loger
import os
loger = app_loger()

file_log = open('webpage_logs/api.txt','a+')


pred_model = predication()

app = Flask(__name__)

@app.route('/api',methods=['POST'])
def stroke_pred():
    content_type = request.headers.get('Content-Type')
    if (content_type == 'application/json'):
        loger.log(file_log,'post request received')
        d = request.get_json()
        test_data= np.array(list(d.values())).reshape(1,-1)
        pred = pred_model.predict(test_data)
        loger.log(file_log,'predication happen from flask web page api')
        if pred == 1:
            return jsonify('Stroke'),200
        else:
            return jsonify('No stroke'),200
        
       
    else:
        loger.log(file_log,'something wrong happned error')
        return jsonify('something went wrong'),404
    

if __name__=="__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)