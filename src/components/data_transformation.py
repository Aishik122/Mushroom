import sys 
from dataclasses import dataclass

import numpy as np 
import pandas as pd
import os 

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,OrdinalEncoder

from src.exception import CustomException
from src.logger import logging 
from src.utils import save_object

@dataclass 
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
    
    def get_data_transformer_object(self):
        try:
            columns = ['class','odor','gill-size','gill-color','stalk-shape','stalk-root','spore-print-color','population']
            pipeline = Pipeline([('last_hope', OrdinalEncoder()),('standard_scaler', StandardScaler())])
            preprocessor=ColumnTransformer([('pipe',pipeline,columns)])
            return preprocessor
            
            
        except Exception as e :
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            logging.info("read train and test data completed")
            
            logging.info("obtaining perprocessing obj")
            preprocessing_obj=self.get_data_transformer_object()
            
            train_arr=preprocessing_obj.fit_transform(train_df)
            test_arr=preprocessing_obj.fit_transform(test_df)
            logging.info('Transformation of train and test done')            
            
            logging.info("Save Preprocessing Object")
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            return (
                train_arr, test_arr,self.data_transformation_config.preprocessor_obj_file_path,
            )
            
        except Exception as e:
            raise CustomException(e,sys)