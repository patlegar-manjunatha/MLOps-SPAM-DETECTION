import logging
import yaml  
import os 
from logging import Logger
from typing import Tuple
import pandas as pd 
from pandas import DataFrame

def initiate_file(params_file_path : str, component_name : str) -> Tuple[dict, Logger]:
    """
Initialize stage-specific configuration and logger.

This function:
- Loads the YAML configuration file
- Extracts parameters for the given pipeline component
- Creates and returns a dedicated logger for that component

Args:
    params_file_path (str): Path to the YAML configuration file.
    component_name (str): Name of the pipeline component
                          (e.g., 'data_ingestion', 'data_preprocessing').

Returns:
    Tuple[dict, Logger]:
        - Dictionary containing configuration for the specified component
        - Logger instance scoped to the specified component

Raises:
    FileNotFoundError: If the params file does not exist.
    yaml.YAMLError: If the YAML file is malformed.
    KeyError: If the component name is not found in the configuration.
"""


    logger = create_logger(component_name)
    try : 
        params = load_yaml(params_file_path, logger)[component_name]
        return params, logger
    except KeyError as e : 
        logger.error('COMPONENT NAME NOT FOUND %s', e)
        raise
    


def create_logger(component_name : str) -> Logger:
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)

    filename = f"{component_name}.log"
    log_file_path = os.path.join(log_dir, filename)

    logger = logging.getLogger(component_name)
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    return logger

def load_yaml(params_file_path : str, logger: Logger) -> dict:
    try: 
        with open(params_file_path, 'r') as file: 
            params = yaml.safe_load(file)
        logger.debug("Parameters retrieved from %s",params_file_path)
        return params
    except FileNotFoundError: 
        logger.error('File not found : %s', params_file_path)
        raise
    except yaml.YAMLError as e : 
        logger.error("YAML error : %s", e)
        raise
    except Exception as e : 
        logger.error("ERROR : %s", e)
        raise


## I/O
def load_data(file_path: str, logger : Logger) -> DataFrame:
    """Loads data from a csv file_path"""
    try : 
        df = pd.read_csv(file_path)
        logger.debug("Data loaded from %s", file_path)
        return df 
    except pd.errors.ParserError as e: 
        logger.error("Failed to parse the CSV file : %s", e)
        raise
    except Exception as e: 
        logger.error("Unexpected error occured while loading the data : %s", e)
        raise


def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str, logger : Logger) -> None: 
    try : 
        raw_data_path = os.path.join(data_path)
        os.makedirs(raw_data_path, exist_ok=True)
        train_data.to_csv(os.path.join(raw_data_path, 'train.csv'), index=False)
        test_data.to_csv(os.path.join(raw_data_path, 'test.csv'), index=False)

        logger.debug("Train and test data saved to %s", raw_data_path)
    except Exception as e: 
        logger.error("Unexpected error occured while saving the data: %s", e)
        raise