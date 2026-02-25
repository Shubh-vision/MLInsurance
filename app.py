import sys
from src.exception import CustomException
from src.logger import logger

if __name__ == "__main__":
    try:
        logger.info("Application started again")
        logger.info("Application started Once again")

    except Exception as e:
        raise CustomException(e, sys)
