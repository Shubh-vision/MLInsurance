import sys
import os
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass



#Decorator
@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    raw_data_path: str = os.path.join("artifacts", "raw.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_Config = DataIngestionConfig()

    def InitiatedDataConfig(self):
        logging.info("Entered the Data ingestion method or component")
        try:
            file_path = os.path.join(os.getcwd(), "notebook", "data", "insurance.csv")
            df = pd.read_csv(file_path)
            logging.info("Read the Dataset as Dataframe")

            os.makedirs(os.path.dirname(self.ingestion_Config.train_data_path), exist_ok=True)
            df.to_csv(self.ingestion_Config.raw_data_path, index=False, header=True)

            logging.info("train test Split initiated...")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_Config.train_data_path, index = False, header = True)
            test_set.to_csv(self.ingestion_Config.test_data_path, index = False, header = True)


            logging.info("Ingestion of Data is complete...")
            return(
                self.ingestion_Config.train_data_path,
                self.ingestion_Config.test_data_path
            )


        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    obj = DataIngestion()
    obj.InitiatedDataConfig()
