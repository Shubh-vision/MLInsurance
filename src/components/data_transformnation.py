import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
import os


@dataclass
class DataTransformationConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    preprocessor_object_file_path : str = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformer_config = DataTransformationConfig()    

    def get_data_transformer_obj(self, input_df):
        try:
            # train_df = pd.read_csv(self.data_transformer_config.train_data_path)
            
            num_col = input_df.select_dtypes(include=['int64', 'float64']).columns
            cat_col = input_df.select_dtypes(include=['object']).columns
            
            num_pipelines = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler(with_mean=False))
                ]
            )

            cat_pipelines = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("OneHotEncoding", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))

                ]
            )

            logging.info("Numerical Col standard scaling completed")
            logging.info("Categorical col endoing completed")

            preprocessorr = ColumnTransformer(
                [
                    ("Num_pipelines", num_pipelines, num_col),
                    ("Cat_pipelines", cat_pipelines, cat_col)
                ]
            )

            return preprocessorr
        
        except Exception as e:
            raise CustomException(e, sys)
        
    
    def initiated_data_transformation(self, train_path, test_path, target_col_name):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test Data completed.")
            logging.info("Obtaining preproessing Object")


            input_feature_train_df = train_df.drop(columns=[target_col_name])
            target_feature_train_df = train_df[target_col_name]

            input_feature_test_df = test_df.drop(columns=[target_col_name])
            target_feature_test_df = test_df[target_col_name]


            preprocessor_obj = self.get_data_transformer_obj(input_feature_train_df)
            logging.info("applying Preprocessing obj on training Dataframe and testing dataframe")



            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)

            train_arr = np.c_[(input_feature_train_arr, np.array(target_feature_train_df))]
            test_arr = np.c_[(input_feature_test_arr, np.array(target_feature_test_df))]

            logging.info("Saved Preprocessing Object")

            save_object(
                file_path=self.data_transformer_config.preprocessor_object_file_path,
                obj = preprocessor_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformer_config.preprocessor_object_file_path
            )


        except Exception as e:
            raise CustomException(e, sys)