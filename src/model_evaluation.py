import os 
import numpy as np 
import pandas as pd 
import pickle
import json 
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from dvclive import Live

from src.helpers import initiate_file
from logging import Logger
from pandas import DataFrame

def load_model(file_path : str, logger: Logger): 
    try: 
        with open(file_path, 'rb') as f: 
            model = pickle.load(f)
        logger.debug("Model loaded from %s", file_path)
        return model
    except FileNotFoundError: 
        logger.error("File not Found : %s", file_path)
        raise
    except Exception as e: 
        logger.error("Unexpected error occured during model loading : %s", e)
        raise 

def load_data(file_path : str, logger: Logger) -> DataFrame:
    try : 
        df = pd.read_csv(file_path)
        logger.debug("Data loaded from %s", file_path)
        return df 
    except pd.errors.ParserError as e: 
        logger.error("Data loaded from %s", file_path)
        raise 
    except Exception as e: 
        logger.error("Unexpected error occured during data loading : %s", e)
        raise 

def evaluate_model(clf, X_test: np.ndarray, y_test : np.ndarray, logger : Logger) -> dict: 
    try: 
        y_pred_proba = clf.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba > 0.25).astype(int)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)

        metrics_dict = {
            'accuracy' : accuracy, 
            'precision' : precision, 
            'recall' : recall,
            'auc' : auc
        }
        logger.debug("Model evaluation metrics calculated")
        return metrics_dict
    except Exception as e: 
        logger.error('Error during model evaluation: %s', e)
        raise 

def save_metrics(metrics : dict, file_path : str, logger : Logger) -> None: 
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, 'w') as f: 
            json.dump(metrics, f, indent=4)
        
        logger.debug("Metrics saved to %s", file_path)
    except Exception as e : 
        logger.error("Error occured while saving the metrics : %s", e)
        raise 

def main(params : dict, logger : Logger): 
    try : 
        clf = load_model(params['model_file_path'], logger=logger)
        test_data = load_data(file_path=params['test_data_path'], logger=logger)

        X_test = test_data.iloc[:, :-1].values
        y_test = test_data.iloc[:, -1].values

        metrics = evaluate_model(clf, X_test, y_test, logger=logger)

        with Live(save_dvc_exp=True) as live:
            live.log_metric('accuracy',metrics['accuracy'])
            live.log_metric('precision', metrics['precision'])
            live.log_metric('recall', metrics['recall'])

            live.log_params(clf.get_params())
        save_metrics(metrics, params['metrics_json_output'], logger=logger) 
        logger.debug('Model evalution stage ended')
    except Exception as e: 
        logger.error("Failed to complete the model evaluation process : %s", e)
        raise   


if __name__ == '__main__': 
    params, logger = initiate_file(params_file_path='params.yaml', component_name='model_evaluation')
    main(params=params, logger=logger)