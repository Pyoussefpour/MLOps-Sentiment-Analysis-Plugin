import pandas as pd
import os
import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import logging

logger = logging.getLogger("data_preprocessing")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler("preprocessing_error.log")
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def preprocess_comment(comment: str) -> str:
    try:
        comment = comment.lower()
        comment = re.sub(r'[^A-Za-z0-9\s!?.,]', '', comment)
        comment = re.sub(r'\s+', ' ', comment)
        comment = comment.strip()
        comment = re.sub(r'\n', ' ', comment)
        stop_words = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}
        comment = ' '.join([word for word in comment.split() if word not in stop_words])

        # Lemmatize the words
        lemmatizer = WordNetLemmatizer()
        comment = ' '.join([lemmatizer.lemmatize(word) for word in comment.split()])

        return comment
    except Exception as e:
        logger.error(f"Error in preprocessing comment: {e}")
        return comment


def normalize_text(df):
    """Apply preprocessing to the text data in the dataframe."""
    try:
        df['clean_comment'] = df['clean_comment'].apply(preprocess_comment)
        logger.debug('Text normalization completed')
        return df
    except Exception as e:
        logger.error(f"Error during text normalization: {e}")
        raise

def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    """Save the processed train and test datasets."""
    try:
        interim_data_path = os.path.join(data_path, 'interim')
        logger.debug(f"Creating directory {interim_data_path}")
        
        os.makedirs(interim_data_path, exist_ok=True)  # Ensure the directory is created
        logger.debug(f"Directory {interim_data_path} created or already exists")

        train_data.to_csv(os.path.join(interim_data_path, "train_processed.csv"), index=False)
        test_data.to_csv(os.path.join(interim_data_path, "test_processed.csv"), index=False)
        
        logger.debug(f"Processed data saved to {interim_data_path}")
    except Exception as e:
        logger.error(f"Error occurred while saving data: {e}")
        raise

def main():
    try:
        logger.debug("Starting data preprocessing...")
        
        # Fetch the data from data/raw
        train_data = pd.read_csv('./data/raw/train.csv')
        test_data = pd.read_csv('./data/raw/test.csv')
        logger.debug('Data loaded successfully')

        # Preprocess the data
        train_processed_data = normalize_text(train_data)
        test_processed_data = normalize_text(test_data)

        # Save the processed data
        save_data(train_processed_data, test_processed_data, data_path='./data')
    except Exception as e:
        logger.error('Failed to complete the data preprocessing process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()
    
