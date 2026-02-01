import pandas as pd 
import os 
from sklearn.feature_extraction.text import TfidfVectorizer

from src.helpers import initiate_file

from logging import Logger
from pandas import DataFrame
from typing import Tuple

def load_data(file_path: str, logger : Logger) -> DataFrame:
    """Loads data from a csv file_path"""
    try : 
        df = pd.read_csv(file_path)
        df.fillna("", inplace=True)
        logger.debug("Data loaded and NaNs filled from %s", file_path)
        return df 
    except pd.errors.ParserError as e: 
        logger.error("Failed to parse the CSV file : %s", e)
        raise
    except Exception as e: 
        logger.error("Unexpected error occured while loading the data : %s", e)
        raise

def apply_tfidf(train_data: DataFrame, test_data: DataFrame, params : dict, logger : Logger) -> Tuple[DataFrame, DataFrame]:
    try : 
        tfidf = TfidfVectorizer(**params['tf_idf'])

        X_train = train_data[params['dependent']]
        y_train = train_data[params['target']]
        X_test = test_data[params['dependent']]
        y_test = train_data[params['target']]

        X_train_bow = tfidf.fit_transform(X_train)
        X_test_bow = tfidf.transform(X_test)

        train_df = pd.DataFrame(X_train_bow.toarray())
        train_df[params['target']] = y_train

        test_df = pd.DataFrame(X_test_bow.toarray())
        test_df[params['target']] = y_test

        logger.debug('tfidf applied and data transformed')
        return train_df, test_df
    except Exception as e : 
        logger.error("Error during BOW transformation : %s", e)
        raise 

def save_data(df: DataFrame, file_path : str, logger : Logger) -> None: 
    try: 
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path, index=False)
        logger.debug('Data saved to %s', file_path)
    except Exception as e : 
        logger.error("Unexpected error occurred while saving the data : %s", e)
        raise 

def main(params : dict, logger : Logger) -> None: 
    try : 
        train_data = load_data(params['train_data_path'], logger= logger)

        test_data = load_data(params['test_data_path'],logger= logger)

        train_data, test_data = apply_tfidf(train_data=train_data, test_data=test_data, params=params, logger=logger) 

        save_data(train_data, file_path=os.path.join("./data", 'processed', 'train_tfidf.csv'), logger=logger)
        
        save_data(test_data, file_path=os.path.join("./data", 'processed', 'test_tfidf.csv'), logger=logger)

    except Exception as e : 
        logger.error('Failed to complete the feature engineering process : %s', e)
        raise


if __name__ == "__main__": 
    params, logger = initiate_file(params_file_path='params.yaml', component_name='feature_engineering')
    main(params, logger)



