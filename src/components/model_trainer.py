from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        r2_score
)
import os
import sys
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')


class ModelTrainer:
    def __init__(self) -> None:
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info('Splitting Training and Test Input Data')
            x_train, y_train, x_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )
            models = {
                "Random Forest": RandomForestClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
            }
            params = {
                "Decision Tree": {
                    'criterion': [ 'gini', 'log_loss', 'entropy'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest": {
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],

                    'max_features':['sqrt','log2',None],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
            }
            model_report: dict = evaluate_model(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
                                                models=models,params=params)
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]
            if best_model_score < 0.6:
                raise CustomException("No Best Model Found")
            logging.info(f"Best found model on both training and testing dataset")
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            predicted = best_model.predict(x_test)
            accuracy = accuracy_score(y_test, predicted)
            # precision = precision_score(y_test, predicted)
            # recall = recall_score(y_test, predicted)
            # r2 = r2_score(y_test, predicted)
            return accuracy
        except Exception as e:
            raise CustomException(e, sys)
