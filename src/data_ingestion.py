import pandas as pd 
import os 
from sklearn.model_selection import train_test_split

from logging import Logger

from src.helpers import initiate_file


def load_data(data_url: str, logger: Logger) -> pd.DataFrame: 
    """
    Docstring for load_data
    
    :param data_url: Description
    :type data_url: str
    :return: Description
    :rtype: DataFrame
    """

    try: 
        df = pd.read_csv(data_url)
        logger.debug('Data loaded from %s', data_url)
        return df 
    except pd.errors.ParserError as e : 
        logger.error("Failed to parse the CSV file : %s", e)
        raise
    except Exception as e : 
        logger.error("Unexcepted error occured while loading the data : %s", e)
        raise


def preprocess_data(df: pd.DataFrame, logger: Logger, params : dict) -> pd.DataFrame: 
    """Preprocessing Stage"""
    try : 
        target = params['feature_names']['target']
        dependent = params['feature_names']['dependent']
        df = df.loc[:, [target, dependent]]
        logger.debug('Data Preprocessing completed')
        return df 
    except KeyError as e : 
        logger.error("Missing column in the dataframe: %s", e)
        raise
    except Exception as e : 
        logger.error("Unexcepted error occured during preprocessing : %s", e)
        raise

def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str, logger : Logger) -> None: 
    try : 
        raw_data_path = os.path.join(data_path, 'raw')
        os.makedirs(raw_data_path, exist_ok=True)
        train_data.to_csv(os.path.join(raw_data_path, 'train.csv'), index=False)
        test_data.to_csv(os.path.join(raw_data_path, 'test.csv'), index=False)

        logger.debug("Train and test data saved to %s", raw_data_path)
    except Exception as e: 
        logger.error("Unexpected error occured while saving the data: %s", e)
        raise

def main(): 
    try: 
        data_path = params['data_path']
        df = load_data(data_url=data_path, logger=logger)
        preprocessed_df = preprocess_data(df, logger=logger, params=params)
        train_data, test_data = train_test_split(preprocessed_df, stratify=preprocessed_df[params['feature_names']['target']], **params['train_test_split'])
        
        save_data(train_data=train_data, test_data=test_data, data_path='./data', logger=logger)
    except Exception as e : 
        logger.error('Failed to complete the data ingestion process : %s', e)
        raise

if __name__ == '__main__': 
    params, logger = initiate_file(params_file_path='params.yaml', component_name='data_ingestion')
    main()