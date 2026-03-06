import sys
from src.exception import CustomException
from src.logger import logger
from fastapi import FastAPI
from pydantic import BaseModel,Field, computed_field
from typing import Literal, Annotated
import pickle
from src.pipeline.predict_pipeline import PredictPipeline
import sklearn

if __name__ == "__main__":
    try:
        logger.info("Application started again")
        logger.info("Application started Once again")

    except Exception as e:
        raise CustomException(e, sys)
    

#Import  ML Models

app = FastAPI(title="Insurance Premium Prediction API")

#Pydantic model to validate incoming data

class UserInput(BaseModel):

    age: Annotated[int, Field(..., gt=0, lt=120, description="Age of the Users")]
    sex: Annotated[str, Field(..., description="sex of the Users")]
    bmi: Annotated[float, Field(...,  description="BMI of the Users")]
    children: Annotated[int, Field(..., description="Total Children of the Users")]
    smoker: Annotated[str, Field(..., description="Is user Smokes?")]
    region: Annotated[str, Field(description="Region of the Users")]


#Home route
@app.get("/")
def home():
    return {"message": "API is running successfully"}


#prediction route
@app.post('/predict')
def predict_premium(data: UserInput):
    try:
        pipeline = PredictPipeline()

        # Convert Pydantic model to dictionary
        input_data = data.model_dump()

        prediction = pipeline.predict(input_data)

        return {
            "predicted_premium": float(prediction)
        }

    except Exception as e:
        logger.error("Prediction failed")
        raise CustomException(e, sys)

print(sklearn.__version__)
