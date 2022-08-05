
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.metrics import accuracy_score,classification_report,roc_auc_score

class model_finder:
    def __init__(self,file_object,logger_object):
        self.file_object = file_object
        self.logger_object = logger_object
        self.clf = RandomForestClassifier()


    def get_best_params_for_random_forest(self,train_x,train_y):
        self.logger_object.log(self.file_object,'enter the get_nest_params_for_random_forest method of the method of model_finder class')
        try:
            self.param_grid = {'bootstrap': [True, False],
                                'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
                                'max_features': ['sqrt', 'log2', None],
                                'min_samples_leaf': [1, 2, 4],
                                'min_samples_split': [2, 5, 10],
                                'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]}

            self.grid = RandomizedSearchCV(estimator=self.clf, param_distributions=self.param_grid,n_iter = 10, cv = 3, verbose=2, random_state=42, n_jobs = -1)
            self.grid.fit(train_x,train_y)
            # extracting the best parameters


            self.max_depth=self.grid.best_params_['max_depth']
            self.bootstrap=self.grid.best_params_['bootstrap']
            self.max_features = self.grid.best_params_['max_features']
            self.min_samples_leaf = self.grid.best_params_['min_samples_leaf']
            self.min_samples_split = self.grid.best_params_['min_samples_split']
            self.n_estimators = self.grid.best_params_['n_estimators']

            #creating new model with besr params

            self.clf = RandomForestClassifier(n_estimators=self.n_estimators,
                                               min_samples_leaf=self.min_samples_leaf, max_features=self.max_features, min_samples_split=self.min_samples_split)
            self.clf.fit(train_x,train_y)
            
            self.logger_object.log(self.file_object,'random forest best params ' + str(self.grid.best_params_) + '.exited from tge get_best_params_for_random method from the best_model_finder class')
            return self.clf
        except Exception as e:
            self.logger_object.log(self.file_object,'Exception occured from get_best_params_for_random_forest from mdoel_finder class')
            raise(e)

    
    def get_bast_model(self,x_train,x_test,y_train,y_test):
        self.logger_object.log(self.file_object,'enter get_best_model method of the tuner class')
        try:
            self.clf = self.get_best_params_for_random_forest(x_train,y_train)
            self.predication_random_forest = self.clf.predict(x_test)

            if len(y_test.unique()) == 1:
                self.random_forest_score = accuracy_score(y_test,self.predication_random_forest)
                self.logger_object.log(self.file_object,'accuracy for rf:' + str(self.random_forest_score))
            else:
                self.accu = accuracy_score(y_test,self.predication_random_forest)
                self.random_forest_score = roc_auc_score(y_test,self.predication_random_forest)
                self.logger_object.log(self.file_object,'auc for rf: '+ str(self.random_forest_score) + ' aucuracy score' + str(self.accu))
            return self.clf
        except Exception as e:
            self.logger_object.log(self.file_object,'exception occured in get_best_model from the tuner class')
            raise(e)

