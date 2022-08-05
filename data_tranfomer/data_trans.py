import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE

class data_tranfomer:
    def __init__(self,file_object,logger_object):
        self.file_object= file_object
        self.logger_object = logger_object

    def upsample(self,x,y):
        self.sm = SMOTE(sampling_strategy='minority')

        self.new_x, self.new_y = self.sm.fit_resample(x,y)
        return self.new_x, self.new_y