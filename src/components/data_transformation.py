import sys 
from dataclasses import dataclass

import numpy as np 
import pandas as pd
import os 

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,OrdinalEncoder,LabelEncoder

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
            columns = ['odor','gill-size','gill-color','stalk-shape','stalk-root','spore-print-color','population']
            pipeline = Pipeline([('Odical Encoder', OrdinalEncoder()),('standard_scaler', StandardScaler())])
            preprocessor=ColumnTransformer([('pipe',pipeline,columns)])
            return preprocessor
            
            
        except Exception as e :
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            logging.info("read train and test data completed")
            
            logging.info("Change names for preprocessing")
            train_df.replace({"class":{'e':'edible','p':'poisonous'},
                            "odor":{'a':'almond','l':'anise','c':'creosote','y':'fishy','f':'foul','m':'musty','n':'none','p':'pungent','s':'spicy'},
                            "gill-size":{'b':'broad','n':'narrow'},
                            "gill-color":{'k':'black','n':'brown','b':'buff','h':'chocolate','g':'gray','r':'green','o':'orange','p':'pink','u':'purple','e':'red',
                                            'w':'white','y':'yellow'},
                            "stalk-shape":{'e':'enlarging','t':'tapering'},
                            "stalk-root":{'b':'bulbous','c':'club','u':'cup','e':'equal','z':'rhizomorphs','r':'rooted'},
                            "spore-print-color":{'k':'black','n':'brown','b':'buff','h':'chocolate','r':'green','o':'orange','u':'purple','w':'white','y':'yellow'},
                            "population":{'a':'abundant','c':'clustered','n':'numerous','s':'scattered','v':'several','y':'solitary'}
                            },inplace=True)
            logging.info('train_df replace done')
            test_df.replace({"class":{'e':'edible','p':'poisonous'},
                            "odor":{'a':'almond','l':'anise','c':'creosote','y':'fishy','f':'foul','m':'musty','n':'none','p':'pungent','s':'spicy'},
                            "gill-size":{'b':'broad','n':'narrow'},
                            "gill-color":{'k':'black','n':'brown','b':'buff','h':'chocolate','g':'gray','r':'green','o':'orange','p':'pink','u':'purple','e':'red',
                                            'w':'white','y':'yellow'},
                            "stalk-shape":{'e':'enlarging','t':'tapering'},
                            "stalk-root":{'b':'bulbous','c':'club','u':'cup','e':'equal','z':'rhizomorphs','r':'rooted'},
                            "spore-print-color":{'k':'black','n':'brown','b':'buff','h':'chocolate','r':'green','o':'orange','u':'purple','w':'white','y':'yellow'},
                            "population":{'a':'abundant','c':'clustered','n':'numerous','s':'scattered','v':'several','y':'solitary'}
                            },inplace=True)
            
            
            
            train_df_x=train_df.drop(columns=['class'])
            train_df_y=train_df['class']
            
            test_df_x=test_df.drop(columns=['class'])
            test_df_y=test_df['class']
            logging.info("Split done for transformation")
            
            logging.info("obtaining perprocessing obj")
            preprocessing_obj=self.get_data_transformer_object()
            
            train_arr_x=preprocessing_obj.fit_transform(train_df_x)
            test_arr_x=preprocessing_obj.fit_transform(test_df_x)
            logging.info('Transformation of train/test features done')
            
            train_arr_y=LabelEncoder().fit_transform(train_df_y)
            test_arr_y=LabelEncoder().fit_transform(test_df_y)
            logging.info('Transformation of test done')
            
            train_arr = np.c_[
                train_arr_x, np.array(train_arr_y)
            ]
            test_arr = np.c_[test_arr_x, np.array(test_arr_y)]

            logging.info(f"Saved preprocessing object.")
            
            logging.info("Save Preprocessing Object")
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            print("out test data is \n ******************************** \n", test_arr)
            return (
                train_arr, test_arr,self.data_transformation_config.preprocessor_obj_file_path,
            )
            
        except Exception as e:
            raise CustomException(e,sys)