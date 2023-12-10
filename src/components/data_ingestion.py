import os
import sys
from src.exception import CustomException
from src.logger import logging
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass  # used to create the class variables
#
# from src.components.model_trainer import ModelTrainer
# from src.components.model_trainer import ModelTrainerConfig


@dataclass  # directly define the class variable
class DataIngestionConfig:  # any data / inputs required are probabaly given through this
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")
    raw_data_path: str = os.path.join('artifacts', "raw.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Data Ingestion Started")
        try:
            df = pd.read_csv('notebook/data/churn.csv')
            logging.info('Data Frame Formed With Data')
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info("Train Test Split Started")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info("Train and Test Dataset Saved")
            logging.info("Data Ingestion Completed")
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data)

    # modeltrainer = ModelTrainer()
    # print(modeltrainer.initiate_model_trainer(train_arr, test_arr))