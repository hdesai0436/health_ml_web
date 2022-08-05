import pickle
import os
import shutil

class file_operation:
    def __init__(self,file_object,logger_object):
        self.file_object = file_object
        self.logger_object = logger_object
        self.model_dir = 'models/'

    def save_model(self,model,file_name):
        self.logger_object.log(self.file_object,'enter save_model method from file_operation')
        try:
            path = os.path.join(self.model_dir,file_name)
            if os.path.isdir(path):
                shutil.rmtree(self.model_dir)
                os.makedirs(path)
            else:
                os.makedirs(path)
            with open(path+'/' + file_name+'.sav','wb') as f:
                pickle.dump(model,f)
            self.logger_object.log(self.file_object,'model file' + file_name + ' saved. exited the save_model of the model_finder class')

            return 'success'
        except Exception as e:
            self.logger_object.log(self.file_object,'exception occured in the save_model method from the file_opeation class' + str(e))

    def load_model(self,filename):
        self.logger_object.log(self.file_object,'enterd the load method of the opeation class')

        try:
            with open(self.model_dir+filename + '/' + filename + '.sav', 'rb') as f:
                self.logger_object.log(self.file_object,'model is loaded from load_model from file_opeation class')
                return pickle.load(f)

        except Exception as e:
            self.logger_object.log(self.file_object,'exception has been occured from load_model method from the file_opeation class', + str(e))


