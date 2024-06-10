from typing import Tuple
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression


if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer


@transformer
def train_model(df: pd.DataFrame, **kwargs) -> Tuple[BaseEstimator, BaseEstimator]:
    features = ['PULocationID', 'DOLocationID']
    target = 'duration'

    # create an instance of the dictionary vectorizer
    dv = DictVectorizer() 

    # convert the categorical variables to a dictionary
    train_dicts = df[features].to_dict(orient = 'records')
    
    # create training data for sklearn model
    X_train = dv.fit_transform(train_dicts)
    y_train = df[target]

    # Linear regression model
    model = LinearRegression()

    # fit data to model
    model.fit(X_train, y_train)
    print(f'The intercept of the model is: {model.intercept_:.2f}')
    
    return dv, model
