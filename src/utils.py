import os
import sys
import numpy as np
import pandas as pd
import dill
from src.exception import CustomException
from src.logger import logging
from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        r2_score
)
from sklearn.model_selection import GridSearchCV

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)

def evaluate_model(x_train,y_train,x_test,y_test,models,params):
    try:
        report = {}
        for i in range(len(list(models))):
            model = list(models.values())[i]
            param = params[list(models.keys())[i]]
            gs =GridSearchCV(model, param, cv=3)
            gs.fit(x_train,y_train)
            model.set_params(**gs.best_params_)
            model.fit(x_train, y_train)
            y_train_pred = model.predict(x_train)
            y_test_pred = model.predict(x_test)
            train_model_score_1 = accuracy_score(y_train, y_train_pred)
            train_model_score_2 = precision_score(y_train, y_train_pred)
            train_model_score_3 = recall_score(y_train, y_train_pred)
            train_model_score_4 = r2_score(y_train, y_train_pred)
            test_model_score_1 = accuracy_score(y_test, y_test_pred)
            # test_model_score_2 = precision_score(y_test, y_test_pred)
            # test_model_score_3 = recall_score(y_test, y_test_pred)
            # test_model_score_4 = r2_score(y_test, y_test_pred)
            report[list(models.keys())[i]] = test_model_score_1 #, test_model_score_2, test_model_score_3, test_model_score_4
        return report
    except Exception as e:
        raise CustomException(e, sys)
