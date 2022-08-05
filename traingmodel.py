from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from application_logs.loger import app_loger
from data_preprocessing.preprocessing import Preprocessor
from data_ingestion.data_loader import Data_Getter
from best_model_finder.tuner import model_finder
from data_tranfomer.data_trans import data_tranfomer
from file_operation.file_methods import file_operation
import numpy as np

class Train_mdoel:
    def __init__(self):
        self.log_writer = app_loger()
        self.file_object = open('traning_logs/modeltraning.txt', 'a+')
        self.scaler = StandardScaler()

    def traning_model(self):
        self.log_writer.log(self.file_object,'traning start')
        try:
            data_getter = Data_Getter(self.file_object,self.log_writer)
            data = data_getter.get_data()

            "doing the data preprocessing"
            preprocessor = Preprocessor(self.file_object,self.log_writer)
            new_data = preprocessor.handle_categorical_feature(data)
            new_data= preprocessor.remove_columns(new_data,['id'])
           
            X,Y = preprocessor.separate_label_feature(new_data,'stroke')
            is_null_presents = preprocessor.is_null_present(X)
            if (is_null_presents):
                X = preprocessor.replace_null_values(X)
            
            dat_tran = data_tranfomer(self.file_object,self.log_writer)

            #upsample training data
            X,Y= dat_tran.upsample(X,Y)

            #split the dataset
            x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=101)

            #apply scaler to the data
            x_train = self.scaler.fit_transform(x_train)
            x_test = self.scaler.transform(x_test)
           
            

            #fit the data in the model

            model = model_finder(self.file_object,self.log_writer)
            best_model = model.get_bast_model(x_train,x_test,y_train,y_test)
            file_op = file_operation(self.file_object,self.log_writer)
            save_model = file_op.save_model(best_model,'random_forest')
            self.log_writer.log(self.file_object,'end of traning')


            
        except Exception as e:
            self.log_writer.log(self.file_object, 'Unsuccessful End of Training')
            raise(e)




            





