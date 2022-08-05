from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer


class Preprocessor:

    def __init__(self,file_object, logger_object):
        self.file_object = file_object
        self.logger_object = logger_object

    def remove_columns(self,data,columns):

        self.logger_object.log(self.file_object, 'enter the remove_columns methods of the preprocessor class')
        self.data = data
        self.columns = columns

        try:
            self.useful_data = self.data.drop(labels=self.columns,axis=1)
            self.logger_object.log(self.file_object,'columns removed successful. exited the removel_columns method of the preprocessor class')
           
            return self.useful_data
        except Exception as e:
            self.logger_object.log(self.file_object,'exception ocuured in the removel_columns method of the proprocessor class :' + str(e))

    def is_null_present(self,data):
        self.logger_object.log(self.file_object,'enter is_null_present in the method of the preprocessor class')
        self.null_present = False
        try:
            self.null_count = data.isna().sum()
            for i in self.null_count:
                if i > 0:
                    self.null_present=True
                    break
            if (self.null_present):
                dataframe_with_null = pd.DataFrame()
                dataframe_with_null['columns'] = data.columns
                dataframe_with_null['missing_values_count'] = np.asarray(data.isna().sum())
                # dataframe_with_null.to_csv('preprocess_data/null_val.csv')
            self.logger_object.log(self.file_object,'findinf missing values is a success. data written to null values in the files. exited is_null_present method in the preprocessing')

            return self.null_present
        except Exception as e:
            self.logger_object.log(self.file_object,'exception ocuured in the is_null_present method of the preprocessing')
            raise(e)

    def replace_null_values(self,data):
        self.logger_object.log(self.file_object,'enter in the replace_null values from prerpocessing class')
        self.data = data
        try:
            imputer = KNNImputer(n_neighbors=3,weights='uniform',missing_values=np.nan)
            self.new_array = imputer.fit_transform(self.data)
            self.new_data = pd.DataFrame(data=self.new_array,columns=self.data.columns)
            self.logger_object.log(self.file_object,'replave missing value from the dataset.exited from the replace_null_values from the preprocesisng class')
            return self.new_data

        except Exception as e:
            self.logger_object.log(self.file_object,'exception occured in the replace_null_vales method from the propressor class')
            raise(e)



    def separate_label_feature(self,data,label_column_name):
        self.logger_object.log(self.file_object,'enter in the separate_label_feature method of the preprocessing class')
        try:
            self.X = data.drop(columns=label_column_name,axis=1)
            self.Y = data[label_column_name]

            
            self.logger_object.log(self.file_object,'label separation successful.Exited from the separate_label_feature method  from the preprocessing class')
            return self.X, self.Y

        except Exception as e:
            self.logger_object.log(self.file_object,'exception ocuured in the separate_label_feature method in the preprocesisng class: ' + str(e))

    
    
    def handle_categorical_feature(self,data):
        self.logger_object.log(self.file_object,'enter in the handle_categorical_feature method in the preprocessing class')
        
        self.data= data
        

        try:
            self.category_col_name = self.data.select_dtypes(include=['object']).columns
            self.new_data = pd.get_dummies(data=self.data,columns=self.category_col_name)
            
            self.logger_object.log(self.file_object,'successful categorized features. exited handle_categorical_feature method from the preprocessing class')
            return self.new_data
        except Exception as e:
            self.logger_object.log(self.file_object,'exception occured in the handle_categorical_feature method in the preproceessing class :' + str(e))


    def label_endoer(self,data,columns_name):
        self.logger_object.log(self.file_object,'enter in the label encode method from the proprecession class')
        self.data = data
        self.col_name = columns_name
        try:
            self.lb = LabelEncoder()
            self.data[self.col_name] = self.lb.fit_transform(self.data[self.col_name])
            self.logger_object.log(self.file_object,'successfuly label encoder feature. exited from the label_encoder method from the preprocessing class')
            return self.data

        except Exception as e:
            self.logger_object.log(self.file_object,'Exception occured from the label_encoder method from the preprocessing class : ' + str(e))



    

    