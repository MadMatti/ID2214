import pandas as pd
import numpy as np

from data import load_data, get_mol, feature_extraction, data_cleaning

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.utils import resample

from sklearn.ensemble import RandomForestClassifier

R = 42

def split(df, which='train'):
    # split into train and test
    y = df['ACTIVE']
    X = df.drop('ACTIVE', axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=R, stratify=y)

    if which == 'train': return X_train, y_train
    elif which == 'test': return X_test, y_test

def oversampling(Xy_train):
    X_train, y_train = Xy_train

    X = pd.concat([X_train, y_train], axis=1)
    yes = X[X['ACTIVE'] == 1.0]
    no = X[X['ACTIVE'] == 0.0]

    yes_upsampled = resample(yes, replace=True, n_samples=len(no), random_state=R)
    upsampled = pd.concat([yes_upsampled, no])
    y_train_up = upsampled['ACTIVE']
    X_train_up = upsampled.drop('ACTIVE', axis=1)
    
    return X_train_up, y_train_up


def transform(Xy_train):
    X_train, y_train = Xy_train

    num_features = X_train.select_dtypes(include=['float64']).columns
    cat_features = X_train.select_dtypes(include=['object', 'bool']).columns
    categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value="missing")),
                                              ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))])
    numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),
                                          ('pca', PCA(n_components=None))])
    preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, num_features), 
                                    ('cat', categorical_transformer, cat_features)])
    
    return preprocessor

def modelling(preprocessor, Xy_train):
    X_train, y_train = Xy_train
    
    forest = Pipeline(steps=[('preprocessor', preprocessor),
                                ('scaler', StandardScaler()),
                                ('classifier', RandomForestClassifier(random_state=R))])

    forest.fit(X_train, y_train)
    return forest

def test_model(model, Xy_test):
    X_test, y_test = Xy_test

    y_pred_prob = model.predict_proba(X_test)[::,1]
    auc = roc_auc_score(y_test, y_pred_prob)
    return auc





if __name__ == "__main__":
    df = load_data()
    df_feat = feature_extraction(get_mol(df))
    df_clean = data_cleaning(df_feat)
    print("AUC score base")
    print(test_model(modelling(transform(split(df_clean, 'train')), split(df_clean, 'train')), split(df_clean, 'test')))
    print("AUC score oversampled")
    print(test_model(modelling(transform(oversampling(split(df_clean, 'train'))), oversampling(split(df_clean, 'train'))), split(df_clean, 'test')))
