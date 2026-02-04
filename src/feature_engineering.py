import pandas as pd 
import os 
from sklearn.feature_extraction.text import TfidfVectorizer

from src.helpers import initiate_file, load_data, save_data

from logging import Logger
from pandas import DataFrame
from typing import Tuple


def apply_tfidf(train_data: DataFrame, test_data: DataFrame, params : dict, logger : Logger) -> Tuple[DataFrame, DataFrame]:
    try : 
        tfidf = TfidfVectorizer(**params['tf_idf'])

        X_train = train_data['dependent']
        y_train = train_data['target']
        X_test = test_data['dependent']
        y_test = test_data['target']

        X_train_bow = tfidf.fit_transform(X_train)
        X_test_bow = tfidf.transform(X_test)

        train_df = pd.DataFrame(X_train_bow.toarray())
        train_df['target'] = y_train

        test_df = pd.DataFrame(X_test_bow.toarray())
        test_df['target'] = y_test

        logger.debug('tfidf applied and data transformed')
        return train_df, test_df
    except Exception as e : 
        logger.error("Error during BOW transformation : %s", e)
        raise 


def main(params : dict, logger : Logger) -> None: 
    try : 
        train_data = load_data(params['train_data_path'], logger= logger)

        test_data = load_data(params['test_data_path'],logger= logger)

        train_data, test_data = apply_tfidf(train_data=train_data, test_data=test_data, params=params, logger=logger) 

        save_data(train_data, test_data, logger=logger, data_path="./data/processed")

    except Exception as e : 
        logger.error('Failed to complete the feature engineering process : %s', e)
        raise


if __name__ == "__main__": 
    params, logger = initiate_file(params_file_path='params.yaml', component_name='feature_engineering')
    main(params, logger)



