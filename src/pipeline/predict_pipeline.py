import sys
import pandas as pd 
import numpy as np
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline(object):
    def __init__(self):
        pass 
    def predict(self,features):
        try:
            model_path='artifacts\model.pkl'
            preprocessor_path='artifacts\preprocessor.pkl'
            model=load_object(model_path)
            preprocessor=load_object(preprocessor_path)
            data_scaled=preprocessor.transform(features)
            
            preds=model.predict(data_scaled)[0]
            
            if preds==0:
                return "Edible"
            else:
                return "Poisonous"
            
        except Exception as e:
            raise CustomException(e,sys)
        
        
class CustomData:
    def __init__(self, 
                 odor:str,
                 gill_size:str,gill_color:str,stalk_shape:str,stalk_root:str,spore_print_color:str,population:str):
        self.odor =odor
        self.gill_size =gill_size
        self.gill_color =gill_color
        self.stalk_shape =stalk_shape
        self.stalk_root =stalk_root
        self.spore_print_color =spore_print_color
        self.population =population
        
    def get_data_frame(self):
        try:
            custom_data_input_dict = {
                'odor':[self.odor],'gill-size':[self.gill_size],'gill-color':[self.gill_color],
                'stalk-shape':[self.stalk_shape],'stalk-root':[self.stalk_root],'spore-print-color':[self.spore_print_color],
                'population':[self.population]
            }
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e :
            raise CustomException(e,sys)