from typing import Tuple
from sklearn.base import BaseEstimator
import mlflow
import pickle


EXPERIMENT_NAME = "homework-03-tracking"


#mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_tracking_uri("sqlite:///home/mlflow/mlflow.db")


mlflow.set_experiment(EXPERIMENT_NAME)
mlflow.sklearn.autolog(disable=True)

if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter


@data_exporter
def export_model(settings: Tuple[BaseEstimator, BaseEstimator], **kwargs):
    vectorizer, model = settings

    with mlflow.start_run():

        # Save the vectorizer to a file
        with open("vectorizer.pkl", "wb") as f:
            pickle.dump(vectorizer, f)

        # log vectorizer
        mlflow.log_artifact("vectorizer.pkl", artifact_path="preprocessor")

        # log model
        mlflow.sklearn.log_model(model, artifact_path="models_mlflow")