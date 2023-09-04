import sys 
from dataclasses import dataclass

import numpy as np 
import pandas as pd
import os 

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import ColumnTransformer
from sklearn.preprocessing import StandardScaler 

from src.exception import CustomException
from src.logger import logging 

@dataclass 
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig
    
    def get_data_transformer_object(self):
        try:
            features=['class','cap-shape','cap-surface','cap-color','bruises','odor','gill-attachment','gill-spacing','gill-size','gill-color',
                      'stalk-shape','stalk-root','stalk-surface-above-ring','stalk-surface-below-ring','stalk-color-above-ring',
                      'stalk-color-below-ring','veil-color','ring-number','ring-type','spore-print-color','population,habitat']
            pipeline=Pipeline(
                steps=[
                    ('label_encoder',LabelEncoder()),
                    ('Standard scaling', StandardScaler())
                ]
            )
            logging.info("Label encoding and scaling completed.")

            preprocessor=ColumnTransformer(
                
            )
            return preprocessor
            
            
        except Exception as e :
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_path=pd.read_cav(test_path)
            logging.info("read train and test data completed")
            
            logging.info("obtaining perprocessing obj")
            preprocessor_obj=self.get_data_transformer_object()
            target_column_name="class"
            
        except Exception as e:
            raise CustomException(e,sys)