# from traingmodel import Train_mdoel

# t = Train_mdoel()

# s= t.traning_model()

from file_operation.file_methods import file_operation
from predictfrommodel import predication
import numpy as np
a =predication()


d=  'Male',38,0,0,'Yes','Private','Rural',108.68,32.7,'never smoked'

print(a.predict(d))


