import joblib
import numpy as np
import os
import sys
import pandas as pd
from src.logger import logger
from src.exception import CustomException


class PredictPipeline:
    def __init__(self):
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

            self.model = joblib.load(model_path)
            self.preprocessor = joblib.load(preprocessor_path)

            logger.info("Model and Preprocessor loaded successfully")

        except Exception as e:
            raise CustomException(e, sys)
        
    def predict(self, data):
        try:
            # Convert incoming data into DataFrame
            input_df = pd.DataFrame([data])

            logger.info("Input DataFrame created")

            # Step 1: Transform
            transformed_data = self.preprocessor.transform(input_df)
            logger.info("Data transformed successfully")

            # Step 2: Predict
            prediction = self.model.predict(transformed_data)
            logger.info("Prediction completed")
            return prediction[0]

        except Exception as e:
            raise CustomException(e, sys)