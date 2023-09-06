import os
import sys
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from src.pipeline.training_pipeline import evaluate_model

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")
    
class ModelTrainer:
    def __init__(self) -> None:
        self.model_trainer_config = ModelTrainerConfig()
        
    def initiate_model_trainer(self, train_features,test_features):
        try:
            logging.info("Splitting train and test data")
            X_train, y_train, X_test, y_test = (
                train_features[:,:-1],
                train_features[:,-1],
                test_features[:,:-1],
                test_features[:,-1],
            )
            
            models = {
                "Logistic Regression": LogisticRegression(),
                "Decision Tree": DecisionTreeClassifier(),
                "Random Forest Classifier": RandomForestClassifier(),
                "Support Vector Classifier": SVC(),               
            }
            
            params = {
                "Logistic Regression": {},
                "Decision Tree": {
                    "max_depth": [3, 5, 7],
                    "criterion": ['entropy', 'gini'],
                    "splitter": ['best', 'random'],
                    "max_features": ['sqrt','log2']                    
                },
                "Random Forest Classifier": {
                    "criterion": ['entropy', 'gini', 'log_loss'],
                    "max_depth": [3, 5, 7],
                    "max_features": ['sqrt','log2'],
                    "n_estimators": [8,16,32,64,128,256]                                                     
                },
                "Support Vector Classifier":{
                    "kernel": ['linear', 'poly', 'rbf', 'sigmoid'],
                    "gamma": ['scale', 'auto']
                }              
            }
            
            model_report: dict = evaluate_model(TrainFeatures=X_train, TrainTarget=y_train,
                                                TestFeatures=X_test, TestTarget=y_test,
                                                models=models, params=params)
            
            logging.info("model hyperparameter tuning done")
            logging.info("model training complete")
            
            # to get best model score from dictionary
            best_model_score = max(sorted(model_report.values()))
        
            # to get best model name from dictionary
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            best_model = models[best_model_name]
            
            if best_model_score<0.6:
                raise CustomException("No Best Model Found", sys)

            logging.info("Best model found on both train and test dataset : {} with accracy score : {}".format(best_model, 
                                                                                                               best_model_score))
            
            save_object(file_path=self.model_trainer_config.trained_model_file_path,
                        obj=best_model)
            
            logging.info("Using the best model found to predict on test data")

            predicted = best_model.predict(X_test)
            
            Accuracy_Score = accuracy_score(y_test, predicted)
            logging.info("Prediction result on test data : Accuracy Score -> {}".format(Accuracy_Score))
            
            return Accuracy_Score, best_model
            
        except Exception as e:
            raise CustomException(e,sys)