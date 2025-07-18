import numpy as np
import pandas as pd
import os
import pickle
import yaml
import logging
import lightgbm as lgb
from sklearn.feature_extraction.text import TfidfVectorizer


logger = logging.getLogger("model_builder")
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler("model_builder_error.log")
file_handler.setLevel('ERROR')

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_params(params_path: str) -> pd.DataFrame:
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug(f"Params loaded from {params_path}")
        return params
    except FileNotFoundError:
        logger.error(f"File not found: {params_path}")
        raise 
    except yaml.YAMLError:
        logger.error("YAML Error %s", e)
        raise 
    except Exception as e:
        logger.error("Error %s", e)
        raise 
    

def load_data(data_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(data_path)
        df.fillna("", inplace=True)
        logger.debug(f"Data loaded from {data_path}")
        return df
    except pd.errors.EmptyDataError as e:
        logger.error(f"failed to parse the CSV file: {e}")
        raise
    except Exception as e:
        logger.error(f"Error: {e}")
        raise
    

def vectorize_text(train_data: pd.DataFrame, max_features: int, n_gram: tuple) -> tuple:
    try:
        vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=n_gram)
        X_train = train_data['clean_comment'].values
        y_train = train_data['category'].values


        X_train_vectorized = vectorizer.fit_transform(X_train)

        logger.debug(f"TFIDF Transformation complete. Train shape: {X_train_vectorized.shape}")


        #save the vectorizer in the root dir
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'../tfidf_vectorizer.pkl'), "wb") as f:
            pickle.dump(vectorizer, f)

        logger.debug("Vectorizer saved in the root dir")

        return X_train_vectorized, y_train
    except Exception as e:
        logger.error(f"Error-TFIDF: {e}")
        raise


def train_lgbm(X_train: pd.DataFrame, y_train: pd.DataFrame, learning_rate: float, max_depth: int, n_estimators: int) -> lgb.LGBMClassifier:
    try:
        model = lgb.LGBMClassifier(
            objective='multiclass',
            num_class=3,
            metric="multi_logloss",
            is_unbalance=True,
            class_weight="balanced",
            reg_alpha=0.1,  # L1 regularization
            reg_lambda=0.1,  # L2 regularization
            learning_rate=learning_rate,
            max_depth=max_depth,
            n_estimators=n_estimators
        )

        model.fit(X_train, y_train)
        logger.debug("LightGBM model trained successfully")
        return model
    except Exception as e:
        logger.error(f"Error-LightGBM-Training: {e}")
        raise
    

def save_model(model: lgb.LGBMClassifier, model_path: str) -> None:
    try:
        with open(model_path, "wb") as file:
            pickle.dump(model, file)
        logger.debug(f"Model saved in {model_path}")
    except Exception as e:
        logger.error(f"Error-LightGBM-Saving: {e}")
        raise
    

def main():
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.abspath(os.path.join(current_dir, '../'))

        params = load_params(params_path=os.path.join(root_dir, 'params.yaml'))
        max_features = params['model_builder']['max_features']
        n_estimators = params['model_builder']['n_estimators']
        learning_rate = params['model_builder']['learning_rate']
        max_depth = params['model_builder']['max_depth']
        ngram_range = tuple(params['model_builder']['ngram_range'])


        train_data = load_data(os.path.join(root_dir,'data/interim/train_processed.csv'))


        X_train_tfidf, y_train = vectorize_text(train_data, max_features, ngram_range)

        model = train_lgbm(X_train_tfidf, y_train, learning_rate, max_depth, n_estimators)

        save_model(model, os.path.join(root_dir, 'lgbm_model.pkl'))

    except Exception as e:
        logger.error(f"Error-Main-model-training: {e}")
        raise
    

if __name__ == "__main__":
    main()