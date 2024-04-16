import pickle
import os

import config

import pandas as pd
from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer
from feature_engine.outliers import Winsorizer
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler

# PARAMS
TARGET = config.TARGET
FEATURES = config.FEATURES
MODEL_PARAMS = config.MODEL_PARAMS


def load_data():
    # Load datasets, specifying the customer IDs as index
    df_train = pd.read_csv('data/train_set.csv', index_col='mk_CurrentCustomer')
    df_customer_country = pd.read_csv('data/customer_country.csv', index_col='mk_CurrentCustomer')
    # Convert date columns to date specifying the format (it's much faster when specifying it)
    df_train['ScoreDate'] = pd.to_datetime(df_train['ScoreDate'], format='%d/%m/%Y %H:%M')
    # Merging the datasets
    df = df_train.join(df_customer_country, how='left')
    return df


def train_model(df, config):
    # Split features & target
    X, y = df[config.FEATURES], df[config.TARGET]
    # Preprocessor
    winsorizer = Winsorizer(capping_method='quantiles', tail='both', fold=0.0001)
    preprocessor = ColumnTransformer([
        ('selected_features', winsorizer, FEATURES)
    ])
    # Resampler
    rus = RandomUnderSampler(random_state=42)
    # Classifier with tuned hyperparams (from PyCaret - using Bayesian search - 100 iterations)
    model = CatBoostClassifier(**config.MODEL_PARAMS, task_type='GPU', devices='0:1', silent=False)
    # Model pipeline
    model_pipeline = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('resampler', rus),
        ('clf', model)
    ])
    # Train model
    model_pipeline.fit(X,y)
    return model_pipeline


def save_model(model, filepath='models/model.pkl'):
    with open(filepath, 'wb') as model_file:
        pickle.dump(model, model_file)


if __name__ == '__main__':
    df = load_data()
    model = train_model(df, config)
    save_model(model)
