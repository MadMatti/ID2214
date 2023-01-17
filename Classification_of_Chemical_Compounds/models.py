from os import linesep
import pandas as pd
import numpy as np

from data import *

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report, roc_curve, RocCurveDisplay, auc
from sklearn.utils import resample
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import shuffle
from sklearn.model_selection import cross_val_score, cross_val_predict

from skopt import BayesSearchCV

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline, make_pipeline

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC 
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

import warnings
warnings.filterwarnings("ignore")



R = 42
AUC_models = {name: [] for name in ['Random Forest', 'Extreme Random Forest', 'Gradient Boosting', 'Logistic Regression', 'Bayes']}

def split(df):
    # split into train and test
    y = df['ACTIVE']
    X = df.drop('ACTIVE', axis=1)
    X_train, y_train = shuffle(X, y, random_state=R) # R

    return X_train, y_train

def oversampling(Xy_train):
    X_train, y_train = Xy_train

    X = pd.concat([X_train, y_train], axis=1)
    yes = X[X['ACTIVE'] == 1.0]
    no = X[X['ACTIVE'] == 0.0]

    yes_upsampled = resample(yes, replace=True, n_samples=len(no)) # R
    upsampled = pd.concat([yes_upsampled, no])
    y_train_up = upsampled['ACTIVE']
    X_train_up = upsampled.drop('ACTIVE', axis=1)
    
    return X_train_up, y_train_up

def undersampling(Xy_train):
    X_train, y_train = Xy_train

    X = pd.concat([X_train, y_train], axis=1)
    yes = X[X['ACTIVE'] == 1.0]
    no = X[X['ACTIVE'] == 0.0]

    no_downsampled = resample(no, replace=False, n_samples=len(yes)) # R
    downsampled = pd.concat([yes, no_downsampled])
    y_train_down = downsampled['ACTIVE']
    X_train_down = downsampled.drop('ACTIVE', axis=1)
    
    return X_train_down, y_train_down

def syntetic_samples(Xy_train):
    X_train, y_train = Xy_train
    
    sm = SMOTE(k_neighbors=20) # R
    X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)

    return X_train_sm, y_train_sm

def syntetic_under(Xy_train):
    X_train, y_train = Xy_train

    over = SMOTE(sampling_strategy=0.4, k_neighbors=20) # R
    under = RandomUnderSampler(sampling_strategy=0.5) # R

    steps = [('o', over), ('u', under)]
    pipeline = Pipeline(steps=steps)
    X_train_su, y_train_su = pipeline.fit_resample(X_train, y_train)

    return X_train_su, y_train_su



def transform(Xy_train):
    X_train, y_train = Xy_train

    num_features = X_train.select_dtypes(include=['float64']).columns
    cat_features = X_train.select_dtypes(include=['object', 'bool']).columns
    categorical_transformer = Pipeline(steps=[
                                      ('imputer', SimpleImputer(strategy='constant', fill_value="missing")),
                                      ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))])
    numeric_transformer = Pipeline(steps=[
                                  ('imputer', SimpleImputer(strategy='median')),
                                  ('pca', PCA(n_components=0.95))])
    preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, num_features), 
                                    ('cat', categorical_transformer, cat_features)])
    
    return preprocessor

def modelling(preprocessor, Xy_train):
    X_train, y_train = Xy_train
    cv = StratifiedKFold(n_splits=5, shuffle=True)

    models = {}

    # extreme random forest
    extreme_parameters = {
        'extreme__random_state': 20,
        'extreme__n_estimators': 1400,
        'extreme__min_samples_split': 12,
        'extreme__min_samples_leaf': 4,
        'extreme__max_features': 'sqrt',
        'extreme__max_depth': 10,
        'extreme__bootstrap': True
    }

    extreme = Pipeline(steps=[('preprocessor', preprocessor),
                                ('scaler', StandardScaler()),
                                ('smote', SMOTE(n_jobs=-1)),
                                ('extreme', ExtraTreesClassifier(n_estimators=1643, min_samples_split=5, min_samples_leaf=1, 
                                                                     max_features='sqrt', max_depth=275, bootstrap=True, n_jobs=-1))]) # R
    
    extreme.fit(X_train, y_train)
    models["Extreme Random Forest"] = extreme

    return extreme
    #return models


def test_model(preprocessor, Xy_test):
    X_test, y_test = Xy_test
    extreme_auc = []
    extreme_f1 = []

    for i in range(10):
        X_test, y_test = shuffle(X_test, y_test)
        cv = StratifiedKFold(n_splits=5, shuffle=True)
        print(i)

        extreme = Pipeline(steps=[('preprocessor', preprocessor),
                                    ('scaler', StandardScaler()),
                                    ('smote', SMOTE(n_jobs=-1)),
                                    ('extreme', ExtraTreesClassifier(n_estimators=1643, min_samples_split=5, min_samples_leaf=1, 
                                                                     max_features='sqrt', max_depth=275, bootstrap=True, n_jobs=-1))]) # R

        score = (cross_val_score(extreme, X_test, y_test, cv=cv, scoring='roc_auc', n_jobs=-1))
        extreme_auc.append(score.mean())
        print("Extreme AUC")
        print(score.mean(), score.std())
        f1_e = cross_val_score(extreme, X_test, y_test, cv=cv, scoring='f1', n_jobs=-1).mean()
        extreme_f1.append(f1_e)
        print("Extreme F1")
        print(f1_e)


    print("Extreme AUC:")
    print(np.mean(extreme_auc))

    return extreme_auc, extreme_f1


def upsampling(preprocessor, Xy):
    X, y = Xy
    cv = StratifiedKFold(n_splits=5, shuffle=True)

    imba_pipeline = make_pipeline(preprocessor,
                                  StandardScaler(),
                                  SMOTE(random_state=42, n_jobs=-1), 
                                  ExtraTreesClassifier(random_state=13, n_jobs=-1))
    cross_val_score(imba_pipeline, X, y, scoring='roc_auc', cv=cv)

    params_b = {
        'extratreesclassifier__n_estimators': (100, 2500),
        'extratreesclassifier__max_depth': (5, 500),
        'extratreesclassifier__min_samples_split': (2, 15),
        'extratreesclassifier__min_samples_leaf': (1, 30),
        'extratreesclassifier__max_features': ['sqrt', 'log2'],
        'extratreesclassifier__bootstrap': [True, False]
    }

    grid_imba = BayesSearchCV(imba_pipeline, search_spaces=params_b, cv=cv, scoring='roc_auc', verbose=3, n_jobs=-1, n_iter=50)
    grid_imba.fit(X, y)

    print("Best parameters:", grid_imba.best_params_)
    print("Best score:", grid_imba.best_score_)

def predict(model, X,y, features):
    cv_test = StratifiedKFold(n_splits=5, shuffle=True, random_state=(R+1))

    AUC = cross_val_score(model, X, y, cv=cv_test, scoring='roc_auc').mean()
    print("AUC: ", AUC)

    y_pred = cross_val_predict(model, X, y, cv=cv_test, n_jobs=-1, method='predict_proba')
    print(y_pred)
    #print(classification_report(y, y_pred))
    # print(y.shape, y_pred.shape)
    # print(y.shape, y_pred[:,-1].shape)
    auc2 = roc_auc_score(y, y_pred[:,-1])
    print("AUC2: ", auc2)
    fpr, tpr, thresholds = roc_curve(y, y_pred[:,-1])
    roc_auc = auc(fpr, tpr)
    print("fpr", fpr)
    print("tpr", tpr)
    RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc).plot()
    plt.show()


    predictor = model

    df_eval = features

    predictions = predictor.predict_proba(df_eval)

    OUT_FILE = 'Assignment_4/resources/9.txt'

    with open(OUT_FILE, 'w') as f:
        f.write(str(AUC) + linesep)
        for prediction in predictions:
            f.write(str(prediction[1]) + linesep)





    

        
    




if __name__ == "__main__":
    test_file = 'Assignment_4/resources/test_smiles.csv'
    df = load_data()
    features = pd.read_csv('Assignment_4/resources/all_features.csv', index_col=0)
    df_feat = final_selection(data_cleaning(features))
    df_clean = data_cleaning(df_feat)

    '''Use this code to train'''
    #upsampling(transform(split(df_clean)), split(df_clean))

    '''Use this code to test not to train'''
    # auc, f1 = test_model(transform(split(df_clean)), split(df_clean))
    # auc2, f12 = test_model(transform(split(df_clean)), split(df_clean))

    '''Use this code to predict'''
    eval_features = selection_prediction(test_file)
    predict(modelling(transform(split(df_clean)), split(df_clean)), X=split(df_clean)[0], y=split(df_clean)[1], features=eval_features)



