import pandas as pd
from file_operation.file_methods import file_operation
from data_preprocessing.preprocessing import Preprocessor
from application_logs.loger import app_loger
import numpy as np
from sklearn.preprocessing import StandardScaler
class predication:
    def __init__(self):
        self.file_object = open('predication_logs/predication_log.txt','a+')
        self.loger = app_loger()
        self.sc = StandardScaler()

    def predict(self,input):
        file_loader = file_operation(self.file_object,self.loger)
        
        test_input = np.array(input,dtype=object).reshape(1,10)
        ohe_gender = file_loader.load_model('gender')
        ohe_ever_married = file_loader.load_model('ever_married')
        ohe_residence_type = file_loader.load_model('Residence_type')
        ohe_smoking_status = file_loader.load_model('smoking_status')
        ohe_work_type = file_loader.load_model('work_type')
        std = file_loader.load_model('stand_scaler')
        self.loger.log(self.file_object,'all one hot encoding models are loaded')
        
        clf = file_loader.load_model('random_forest')

        self.loger.log(self.file_object,'random forest model is loaded')

        gender_tranform = ohe_gender.transform(test_input[:,0].reshape(1,1))
        ever_married_transform = ohe_ever_married.transform(test_input[:,4].reshape(1,1))
        residence_type = ohe_residence_type.transform(test_input[:,5].reshape(1,1))
        smoking_status_transform = ohe_smoking_status.transform(test_input[:,9].reshape(1,1))
        work_type_transform = ohe_work_type.transform(test_input[:,6].reshape(1,1))
       
        test_input_tranforms = np.column_stack((test_input[:,[1,2,3,7,8]],gender_tranform,ever_married_transform,residence_type,smoking_status_transform,work_type_transform))
        
        self.loger.log(self.file_object,'all test data has been transformed to predict')

        stand = std.transform(test_input_tranforms)
        pred = clf.predict(stand)


        self.loger.log(self.file_object,'predication values for test data is ' + str(pred))
        
        
        return pred



        
        
