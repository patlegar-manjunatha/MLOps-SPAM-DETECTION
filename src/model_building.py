from src.helpers import initiate_file
from logging import Logger
from pandas import DataFrame

import os 
import numpy as np 
import pandas as pd 
import pickle
from sklearn.ensemble import RandomForestClassifier

def load_data(filepath : str, logger : Logger) -> DataFrame: 
    try : 
        df = pd.read_csv(filepath)
        logger.debug('Data loaded from %s with shape %s', filepath, df.shape)
        return df 
    except pd.errors.ParserError as e: 
        logger.error('Failed to parse the CSV File : %s', e)
        raise 
    except Exception as e : 
        logger.error("Unexpected error occured while loading the data : %s", e)
        raise

def train_model(X_train: np.ndarray, y_train: np.ndarray, params : dict, logger: Logger) -> RandomForestClassifier: 
    try : 
        if X_train.shape[0] != y_train.shape[0]: 
            raise ValueError("The number of sample in X_train and y_train must be equal")
        
        logger.debug("Initializing RandomForest model with params : %s", params)

        clf = RandomForestClassifier(**params)
        clf.fit(X_train, y_train)
        logger.debug("Model Training Completed")
        return clf 
    except ValueError as e: 
        logger.error("ValueError during model training : %s",e)
        raise 
    except Exception as e : 
        logger.error("Unexpected error occured in model training stage : %s", e)
        raise


def save_model(model, file_path : str, logger: Logger) -> None: 
    try : 
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, 'wb') as f: 
            pickle.dump(model, f)
        logger.debug("Model saved at %s", file_path)
    except FileNotFoundError as e : 
        logger.error("File path not found : %s", e)
        raise
    except Exception as e: 
        logger.error("Unexpected Error occued while saving the model : %s",e )
        raise

def main(params: dict, logger: Logger) -> None:
    try : 
        train_data = pd.read_csv(params['train_data_path'])
        X_train = train_data.iloc[:, :-1].values
        y_train = train_data.iloc[:, -1].values

        clf = train_model(X_train=X_train, y_train=y_train, params=params['random_forest'], logger=logger)

        model_save_path = 'model/model.pkl'
        save_model(clf, model_save_path, logger=logger)

    except Exception as e: 
        logger.error('Failed to complete the model building process : %s', e)
        raise 


if __name__ == '__main__': 
    params, logger = initiate_file(params_file_path='params.yaml', component_name='model_building')
    main(params=params, logger=logger)