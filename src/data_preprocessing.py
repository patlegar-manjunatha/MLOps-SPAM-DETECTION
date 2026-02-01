import os 
import pandas as pd 
from sklearn.preprocessing import LabelEncoder
import spacy
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from src.helpers import initiate_file

from logging import Logger
from pandas import DataFrame

def transform_text(text : str) -> str: 
    """
    Transforms the input text by converting it to lowercase, tokenizing, removing stopwords and punctuation, and stemming.
    """
    doc = nlp(text)
    transformed_text = [token.lemma_.lower() for token in doc if not token.is_punct and not token.is_stop and token.is_alpha]
    return " ".join(transformed_text)

def preprocess_df(df : DataFrame, params : dict, logger : Logger) -> DataFrame: 
    """
    Preprocesses the DataFrame by encoding the target column, removing duplicates, and transforming the text column.
    """
    try :
        target = params['target']
        text = params['dependent']
        logger.debug("Starting preprocessing for DataFrame")

        encoder = LabelEncoder()
        df[target] = encoder.fit_transform(df[target])
        logger.debug("Target column encoded")

        df = df.drop_duplicates(keep='first')
        logger.debug("Duplicates removed")
        
        df.loc[:, text] = df[text].apply(transform_text)
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
        train_data = pd.read_csv(params['train_data_path'])
        test_data = pd.read_csv(params['test_data_path'])

        train_processed_data = preprocess_df(train_data, params, logger=logger)

        test_processed_data = preprocess_df(test_data, params=params, logger=logger)

        data_path = os.path.join("./data", 'interim')
        os.makedirs(data_path, exist_ok=True)

        train_processed_data.to_csv(os.path.join(data_path, "train_processed.csv"), index=False)

        test_processed_data.to_csv(os.path.join(data_path, 'test_processed.csv'), index=False)

        logger.debug("Processed data saved to %s", data_path)

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

