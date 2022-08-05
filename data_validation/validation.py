from multiprocessing.spawn import import_main_path


import os
import re
import json
import shutil
import pandas as pd
from application_logs.loger import app_loger

class raw_data_validation:
    def __init__(self):
        self.schema_path = 'schema_traning.json',
        self.logger = app_loger()


    def ValuesFromSchema(self):
        try:
            with open(self.schema_path,'r') as f:
                dic = json.load(f)
                f.close()
            columns_names = dic['ColName']
            numbercolumns = dic['NumberofColumns']
            file = open('traning_logs/valuesfromschemalog.txt',"a+")
            message = "numbercolumns:: %s" + numbercolumns
            self.logger.log(file,message)
            file.close()
        except ValueError:
            file = open('traning_logs/valuesfromschemalog.txt',"a+")
            self.logger.log(file,'valueError: value not found inside schema_training.json')
            file.close()
            raise ValueError
        except Exception as e:
            file = open('traning_logs/valuesfromschemalog.txt',"a+")
            self.logger.log(file,str(e))
            file.close()
            raise e
        return columns_names,numbercolumns

    def validateColumnlength(self,numbercolumns):
        try:
            f = open('traning_logs/columnvalidation.txt','a+')
            self.logger.log(f,'column length validation started')
            df = pd.read_csv('dataset/healthcare-dataset-stroke-data.csv')
            if df.shape[1] == numbercolumns:
                pass
            else:
                self.logger.log(f,'invalid columns length for the file')
            f.close()
        except Exception as e:
            f = open('traning_logs/columnvalidation.txt','a+')
            self.logger.log(f,"error occured :: %s" + str(e))
            f.close()
            raise(e)



