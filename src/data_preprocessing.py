import os 
import pandas as pd 
from sklearn.preprocessing import LabelEncoder
import spacy
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from src.helpers import initiate_file, load_data, save_data

from logging import Logger
from pandas import DataFrame

def transform_text(text : str) -> str: 
    """
    Transforms the input text by converting it to lowercase, tokenizing, removing stopwords and punctuation, and stemming.
    """
    doc = nlp(text)
    transformed_text = [token.lemma_.lower() for token in doc if not token.is_punct and not token.is_stop and token.is_alpha or token.is_digit]
    return " ".join(transformed_text)

def preprocess_df(df : DataFrame, logger : Logger) -> DataFrame: 
    """
    Preprocesses the DataFrame by encoding the target column, removing duplicates, and transforming the text column.
    """
    try :
        logger.debug("Starting preprocessing for DataFrame")

        encoder = LabelEncoder()
        df['target'] = encoder.fit_transform(df['target'])
        logger.debug("Target column encoded")
        
        df.dropna(inplace=True)
        df = df.drop_duplicates(keep='first')

        logger.debug("Duplicates removed")
        
        df.loc[:, "dependent"] = df["dependent"].apply(transform_text)
        logger.debug("Text Column transformed")
        return df
    except KeyError as e : 
        logger.error("Column not found : %s", e)
        raise 
    except Exception as e : 
        logger.error("Unexpected error during text normalization : %s", e)
        raise

def main(params : dict, logger : Logger) -> None: 
    try : 
        train_data = load_data(file_path=params['train_data_path'], logger=logger)
        test_data = load_data(file_path=params['test_data_path'], logger=logger)

        train_processed_data = preprocess_df(train_data, logger=logger)

        test_processed_data = preprocess_df(test_data, logger=logger)

        save_data(train_data=train_processed_data, test_data=test_processed_data, data_path="./data/interim", logger=logger)

    except FileNotFoundError as e : 
        logger.error("File not found ; %s", e)
        raise
    except pd.errors.EmptyDataError as e : 
        logger.error('No data : %s', e)
        raise
    except Exception as e : 
        logger.error("Unexpected error occured in preprocessing stage : %s", e)
        raise


if __name__=='__main__':
    try: 
        nlp = spacy.load("en_core_web_sm")
    except OSError: 
        from spacy.cli import download
        download('en_core_web_sm')
        nlp = spacy.load("en_core_web_sm")

    params, logger = initiate_file(params_file_path='params.yaml', component_name='data_preprocessing')

    main(params=params, logger=logger)

