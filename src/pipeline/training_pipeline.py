from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from src.exception import CustomException
import sys 
import os 






def evaluate_model(TrainFeatures, TrainTarget, TestFeatures, TestTarget, models, params):
    try:
        report = {}
        
        for i in range(len(list(models))):
            model = list(models.values())[i]
            para = params[list(models.keys())[i]]
            
            gs = GridSearchCV(model, para, cv=3)
            gs.fit(TrainFeatures, TrainTarget)
            
            model.set_params(**gs.best_params_)            
            model.fit(TrainFeatures, TrainTarget)
            
            y_train_pred = model.predict(TrainFeatures)
            y_test_pred = model.predict(TestFeatures)
            
            train_model_score = accuracy_score(TrainTarget, y_train_pred)
            test_model_score = accuracy_score(TestTarget, y_test_pred)
            
            report[list(models.keys())[i]] = test_model_score
            
        return report
    
    except Exception as e:
        raise CustomException(e,sys)
        

