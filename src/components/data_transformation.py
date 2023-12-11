import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessot.pkl")


class DataTransformation:
    def __init__(self) -> None:
        self.data_tranformation_config = DataTransformationConfig()

    def get_data_transformer_obj(self):
        try:
            numerical_columns = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']
            categorical_columns = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                                   'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                                   'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
                                   'PaymentMethod']
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )
            logging.info("Numerical columns scaling completed")
            cat_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy="most_frequent")),
                    ('onehotencoder', OneHotEncoder()),
                    ('scaler', StandardScaler(with_mean=False))
                ]
            )
            logging.info("Categorical columns encoding completed")
            preprocessor = ColumnTransformer(
                [
                    ('cat_pipeline', cat_pipeline, categorical_columns),
                    ('num_pipeline', num_pipeline, numerical_columns)

                ]
            )
            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Reading train and test data")
            logging.info("Obtaining Preprocessing Object")
            preprocessing_obj = self.get_data_transformer_obj()
            target_column_name = "Churn"
            numerical_columns = ["writing_score", "reading_score"]
            input_feature_train_df = train_df.drop(columns=[target_column_name, 'customerID'], axis=1)
            target_feature_train_df = train_df[target_column_name]
            input_feature_test_df = test_df.drop(columns=[target_column_name, 'customerID'], axis=1)
            target_feature_test_df = test_df[target_column_name]
            logging.info(f"Applying preprocessing object on training dataframe and testing dataframe")
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
            logging.info(f"Applying label encoder object on training dataframe and testing dataframe")
            le = LabelEncoder()
            target_feature_train_df = le.fit_transform(target_feature_train_df)
            target_feature_test_df = le.transform(target_feature_test_df)
            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]
            logging.info(f"Saved preprocessing object")
            save_object(
                file_path=self.data_tranformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            return (
                train_arr,
                test_arr,
                self.data_tranformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            raise CustomException(e, sys)
