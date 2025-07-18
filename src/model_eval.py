import numpy as np
import pandas as pd
import pickle
import logging
import yaml
import mlflow
import mlflow.sklearn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import matplotlib.pyplot as plt
import seaborn as sns
import json
from mlflow.models import infer_signature


logger = logging.getLogger("model_eval")
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler("model_eval_error.log")
file_handler.setLevel('ERROR')

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_data(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        df.fillna("", inplace=True)
        logger.debug(f"Data loaded from {file_path}")
        return df
    except Exception as e:
        logger.error(f"Error-Loading-Data: {e}")
        raise


def load_model(model_path: str):
    try:
        with open(model_path, "rb") as file:
            model = pickle.load(file)
        logger.debug(f"Model loaded from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Error-Loading-Model: {e}")
        raise

def load_vectorizer(vectorizer_path: str):
    try:
        with open(vectorizer_path, "rb") as file:
            vectorizer = pickle.load(file)
        logger.debug(f"Vectorizer loaded from {vectorizer_path}")
        return vectorizer
    except Exception as e:
        logger.error(f"Error-Loading-Vectorizer: {e}")
        raise

def load_params(params_path: str):
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug(f"Params loaded from {params_path}")
        return params
    except Exception as e:
        logger.error(f"Error-Loading-Params: {e}")
        raise

def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray):
    try:
        y_pred = model.predict(X_test)
        class_report = classification_report(y_test, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_test, y_pred)
        logger.debug("Model evaluated successfully")
        return class_report, conf_matrix
    except Exception as e:
        logger.error(f"Error-Evaluating-Model: {e}")
        raise

def graph_confusion_matrix(confusion_matrix: np.ndarray, dataset_name: str):
    try:
        plt.figure(figsize=(10, 7))
        sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.savefig('confusion_matrix.png')


        cm_file_path = f'confusion_matrix_{dataset_name}.png'
        plt.savefig(cm_file_path)
        mlflow.log_artifact(cm_file_path)
        plt.close()

    except Exception as e:
        logger.error(f"Error-Graphing-Confusion-Matrix: {e}")
        raise


def save_model_info(run_id: str, model_path: str, file_path: str):
    try:
        model_info = {
            "run_id": run_id,
            "model_path": model_path,
        }
        with open(file_path, 'w') as file:
            json.dump(model_info, file, indent=4)
        logger.debug(f"Model info saved in {file_path}")
    except Exception as e:
        logger.error(f"Error-Saving-Model-Info: {e}")
        raise
    
    
def main():
    mlflow.set_tracking_uri("http://ec2-16-52-85-241.ca-central-1.compute.amazonaws.com:5000")
    mlflow.set_experiment("dvc-pipeline-model-eval")

    with mlflow.start_run() as run:
        print("Artifact URI:", mlflow.get_artifact_uri())
        try:
            root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
            params = load_params(os.path.join(root_dir, 'params.yaml'))

            for key, value in params.items():
                mlflow.log_param(key, value)

            model = load_model(os.path.join(root_dir, 'lgbm_model.pkl'))
            vectorizer = load_vectorizer(os.path.join(root_dir, 'tfidf_vectorizer.pkl'))

            test_data = load_data(os.path.join(root_dir, 'data/interim/test_processed.csv'))
            X_test_tfidf = vectorizer.transform(test_data['clean_comment'].values)
            y_test = test_data['category'].values

            # Create a DataFrame for signature inference (using first few rows as an example)
            input_example = pd.DataFrame(X_test_tfidf.toarray()[:5], columns=vectorizer.get_feature_names_out()) 

            #Infer the signature
            signature = infer_signature(input_example, model.predict(X_test_tfidf[:5]))  # <--- Added for signature

            mlflow.sklearn.log_model(
                model,
                "lgbm_model",
                signature=signature,  # <--- Added for signature
                input_example=input_example  # <--- Added input example
            )

            # Save model info
            artifact_uri = mlflow.get_artifact_uri()
            model_path = os.path.join(artifact_uri, "lgbm_model")
            save_model_info(run.info.run_id, model_path, 'experiment_info.json')

            # Log the vectorizer as an artifact
            mlflow.log_artifact(os.path.join(root_dir, 'tfidf_vectorizer.pkl'))

            # Evaluate model and get metrics
            report, cm = evaluate_model(model, X_test_tfidf, y_test)

            for label, metrics in report.items():
                if isinstance(metrics, dict):
                    mlflow.log_metrics({
                        f"test_{label}_precision": metrics['precision'],
                        f"test_{label}_recall": metrics['recall'],
                        f"test_{label}_f1-score": metrics['f1-score']
                    })
            
            # Log confusion matrix
            graph_confusion_matrix(cm, "Test Data")

            mlflow.set_tag("model_type", "LightGBM")
            mlflow.set_tag("task", "Sentiment Analysis")
            mlflow.set_tag("dataset", "YouTube Comments")

        except Exception as e:
            logger.error(f"Failed to complete model evaluation: {e}")
            print(f"Error: {e}")
            raise

if __name__ == "__main__":
    main()


