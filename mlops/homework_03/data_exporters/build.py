from sklearn.base import BaseEstimator
import mlflow.sklearn

if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter


@data_exporter
def export_model(settings: Tuple[BaseEstimator, BaseEstimator], **kwargs) 
-> Tuple[BaseEstimator, BaseEstimator]:
    dv, model = settings

    mlflow.sklearn.log_model(model, "lin_reg")
    mlflow.log_artifacts(dv)




