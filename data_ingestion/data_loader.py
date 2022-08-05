import pandas as pd
class Data_Getter:
    def __init__(self, file_object, logger_object):
        self.data = 'dataset/healthcare-dataset-stroke-data.csv'
        self.file_object= file_object
        self.logger_object = logger_object

    def get_data(self):
        """
        method_name = get data
        description: this method read data
        oupput: a pandas dataframe
        
        """

        self.logger_object.log(self.file_object,'enter the get_data method of the data getter class')
        try:
            self.data = pd.read_csv(self.data)
            self.logger_object.log(self.file_object,'data loaded succesfully')
            return self.data
        except Exception as e:
            self.logger_object.log(self.file_object, 'Exception ocuured in the get_data method in the data_getter class: ' + str(e))

