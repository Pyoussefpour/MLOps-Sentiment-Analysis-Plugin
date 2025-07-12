import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import yaml
import logging

logger = logging.getLogger("data_ingestion")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler("error.log")
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_params(params_path: str) -> dict:
    try:
        with open(params_path, "r") as yaml_file:
            params = yaml.safe_load(yaml_file)
        logger.debug(f"Params retrieved: %s", params_path)
        return params
    except FileNotFoundError as e:
        logger.error(f"File not found: %s", e)
        raise 
    except yaml.YAMLError as e:
        logger.error(f"YAML Error: %s", e)
        raise
    except Exception as e:
        logger.error(f"Error: %s", e)
        raise


def load_data(data_url: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(data_url)
        logger.debug("Data load from: %s", data_url)
        return df
    except pd.errors.EmptyDataError as e:
        logger.error(f"File to parse the CSV file: %s", e)
        raise
    except Exception as e:
        logger.error(f"Error: %s", e)
        raise

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df.dropna(inplace=True)
        df.drop_duplicates(inplace=True)
        df = df[df['clean_comment'].str.strip() != ""]
        logger.debug("Data preprocessing completed: Missing values and duplicates removed")
        return df
    
    except KeyError as e:
        logger.error(f"KeyError: %s", e)
        raise
    except Exception as e:
        logger.error(f"Error: %s", e)
        raise

def save_data(train_df: pd.DataFrame, test_df: pd.DataFrame, data_path: str) -> None:
    try:
        raw_data_path = os.path.join(data_path, "raw")
        os.makedirs(raw_data_path, exist_ok=True)

        train_df.to_csv(os.path.join(raw_data_path, "train.csv"), index=False)
        test_df.to_csv(os.path.join(raw_data_path, "test.csv"), index=False)

        logger.debug("Train and test data saved in: %s", raw_data_path)

    except Exception as e:
        logger.error(f"Error: %s", e)
        raise
    

def main() -> None:
    try:
        params = load_params(params_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../params.yaml")))
        test_size = params["data_ingestion"]["test_size"]

        df = load_data(data_url='https://raw.githubusercontent.com/Himanshu-1703/reddit-sentiment-analysis/refs/heads/main/data/reddit.csv')

        final_df = preprocess_data(df)

        train_df, test_df = train_test_split(final_df, test_size=test_size, random_state=42)


        save_data(train_df, test_df, data_path=os.path.abspath(os.path.join(os.path.dirname(__file__), "../data")))

    except Exception as e:
        logger.error(f"Failed to ingest data: %s", e)
        raise
    

if __name__ == "__main__":
    main()
    