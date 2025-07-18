import mlflow
import json
import os
import logging

mlflow.set_tracking_uri("http://ec2-16-52-85-241.ca-central-1.compute.amazonaws.com:5000")

logger = logging.getLogger("register_model")
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler("register_model_error.log")
file_handler.setLevel('ERROR')

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_model_info(file_path: str):
    try:
        with open(file_path, 'r') as file:
            model_info = json.load(file)
        logger.debug(f"Model info loaded from {file_path}")
        return model_info
    except Exception as e:
        logger.error(f"Error-Loading-Model-Info: {e}")
        raise

def register_model(model_name: str, model_info: dict):
    try:
        model_uri = f"runs:/{model_info['run_id']}/lgbm_model"
        print('model_uri', model_uri)

        model_version = mlflow.register_model(model_uri, model_name)

        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage="Staging"
        )

        logger.debug(f"Model registered successfully")
    except Exception as e:
        logger.error(f"Error-Registering-Model: {e}")
        raise


def main():
    try:
        model_info_path = "experiment_info.json"
        model_info = load_model_info(model_info_path)
        model_uri = f"runs:/{model_info['run_id']}/{model_info['model_path']}"
        print('model_uri', model_uri)

        model_name = "sentiment_analysis_yt_comments"
        register_model(model_name, model_info)

    except Exception as e:
        logger.error(f"Error-Registering-Model: {e}")
        raise

if __name__ == "__main__":
    main()
    