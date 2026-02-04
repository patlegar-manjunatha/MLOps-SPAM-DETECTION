import pandas as pd 
import os 
from sklearn.model_selection import train_test_split

from logging import Logger

from src.helpers import initiate_file, load_data, save_data


def preprocess_data(df: pd.DataFrame, logger: Logger, params : dict) -> pd.DataFrame: 
    """Preprocessing Stage"""
    try : 
        target = params['feature_names']['target']
        dependent = params['feature_names']['dependent']
        df = df.loc[:, [target, dependent]]
        df = df.rename(columns={target : 'target', dependent : 'dependent'})
        logger.debug('Data Preprocessing and Rename of columns into target and dependent completed')
        return df 
    except KeyError as e : 
        logger.error("Missing column in the dataframe: %s", e)
        raise
    except Exception as e : 
        logger.error("Unexcepted error occured during preprocessing : %s", e)
        raise

def main(): 
    try: 
        data_path = params['data_path']
        df = load_data(file_path=data_path, logger=logger)
        preprocessed_df = preprocess_data(df, logger=logger, params=params)
        train_data, test_data = train_test_split(preprocessed_df, stratify=preprocessed_df['target'], **params['train_test_split'])
        
        save_data(train_data=train_data, test_data=test_data, data_path='./data/raw', logger=logger)
    except Exception as e : 
        logger.error('Failed to complete the data ingestion process : %s', e)
        raise

if __name__ == '__main__': 
    params, logger = initiate_file(params_file_path='params.yaml', component_name='data_ingestion')
    main()