import pandas as pd
import numpy as np

from data import load_data, get_mol, feature_extraction, data_cleaning, feature_extraction_1

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.utils import resample
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.svm import SVC 
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB



R = 42
AUC_models = {name: [] for name in ['Random Forest', 'Extreme Random Forest', 'Gradient Boosting', 'Logistic Regression', 'Bayes']}

def split(df, which='train'):
    # split into train and test
    y = df['ACTIVE']
    X = df.drop('ACTIVE', axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y) # R

    if which == 'train': return X_train, y_train
    elif which == 'test': return X_test, y_test

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

    over = SMOTE(sampling_strategy=0.1, k_neighbors=20) # R
    under = RandomUnderSampler(sampling_strategy=0.5) # R

    steps = [('o', over), ('u', under)]
    pipeline = Pipeline(steps=steps)
    X_train_su, y_train_su = pipeline.fit_resample(X_train, y_train)

    return X_train_su, y_train_su



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
    cv = StratifiedKFold(n_splits=5, shuffle=True)

    models = {}

    # # random forest
    # forest = Pipeline(steps=[('preprocessor', preprocessor),
    #                             ('scaler', StandardScaler()),
    #                             ('forest', RandomForestClassifier())]) # R

    # forest.fit(X_train, y_train)
    # models["Random Forest"] = forest

    # params_forest = { 
    # 'forest__bootstrap': [True, False],
    # 'forest__max_depth': [10, 20, 30, 40, 45, 50, 55, 60, 65, 70, 75, 80, 90, 100, None],
    # 'forest__max_features': ['sqrt', 'log2'],
    # 'forest__min_samples_leaf': [1, 2, 3, 4],
    # 'forest__min_samples_split': [2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    # 'forest__n_estimators': [200, 600, 800, 850, 875, 900, 950, 975, 1000, 1100, 1200, 1400, 1600, 1800, 2000, 2200, 2400],
    # }

    # forest_search = RandomizedSearchCV(estimator=forest, param_distributions=params_forest, n_iter=10, verbose=1, n_jobs=-1, cv=cv, scoring='roc_auc')
    # forest_search.fit(X_train, y_train)
    # print("Params and score for Random Forest")
    # print(forest_search.best_params_)
    # print(forest_search.best_score_)

    # extreme random forest
    extreme = Pipeline(steps=[('preprocessor', preprocessor),
                                ('scaler', StandardScaler()),
                                ('extreme', ExtraTreesClassifier())]) # R
    
    extreme.fit(X_train, y_train)
    models["Extreme Random Forest"] = extreme

    return models

    # params_extreme = { 
    #     'extreme__bootstrap': [True, False],
    #     'extreme__max_depth': [10, 20, 30, 40, 45, 50, 55, 60, 65, 70, 75, 80, 90, 100, None],
    #     'extreme__max_features': ['sqrt', 'log2'],
    #     'extreme__min_samples_leaf': [1, 2, 3, 4],
    #     'extreme__min_samples_split': [2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    #     'extreme__n_estimators': [200, 600, 800, 850, 875, 900, 950, 975, 1000, 1100, 1200, 1400, 1600, 1800, 2000, 2200, 2400],
    #     'extreme__random_state': [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, None],
    # }

    # extreme_search = RandomizedSearchCV(extreme, param_distributions=params_extreme, n_iter=10, verbose=1, n_jobs=-1, cv=cv, scoring='roc_auc')
    # extreme_search.fit(X_train, y_train)
    # print(extreme_search.best_params_)
    # print(extreme_search.best_score_)

    # SVM
    # svm = Pipeline(steps=[('preprocessor', preprocessor),
    #                         ('scaler', StandardScaler()),
    #                         ('classifier', SVC(random_state=R, probability=True))])
    
    # svm.fit(X_train, y_train)
    # models["SVM"] = svm
    # print("Done")

    # gradient boosting
    gboost = Pipeline(steps=[('preprocessor', preprocessor),
                            ('scaler', StandardScaler()),
                            ('classifier', GradientBoostingClassifier())]) # R
    
    gboost.fit(X_train, y_train)
    models["Gradient Boosting"] = gboost

    # logistic regression
    logreg = Pipeline(steps=[('preprocessor', preprocessor),
                            ('scaler', StandardScaler()),
                            ('classifier', LogisticRegression())]) # R

    logreg.fit(X_train, y_train)
    models["Logistic Regression"] = logreg

    # bayes
    bayes = Pipeline(steps=[('preprocessor', preprocessor),
                            ('scaler', StandardScaler()),
                            ('classifier', GaussianNB())])

    bayes.fit(X_train, y_train)
    models["Bayes"] = bayes

    return models
    # adaboost
    adaboost = Pipeline(steps=[('preprocessor', preprocessor),
                            ('scaler', StandardScaler()),
                            ('classifier', AdaBoostClassifier(random_state=R))])

    adaboost.fit(X_train, y_train)
    models["Adaboost"] = adaboost

    # ensemble
    forest_e = RandomForestClassifier(random_state=R, n_estimators=1600, min_samples_split=11, 
                                        min_samples_leaf=1, max_features='log2', max_depth=100, bootstrap=True)
    
    gradient_e = GradientBoostingClassifier(random_state=R, n_estimators=75, max_depth=5, learning_rate=0.2)
    bayes_e = GaussianNB()
    extreme_forest_e2 = ExtraTreesClassifier(random_state=900, n_estimators=975, min_samples_split=5, 
                                            min_samples_leaf=1, max_features='sqrt', max_depth=40, bootstrap=False)

    estimators = [('forest', forest_e), ('bayes', bayes_e), ('gradient', gradient_e), ('extreme', extreme_forest_e2)]

    ensemble = Pipeline(steps=[('preprocessor', preprocessor), ('standardscaler', StandardScaler()), 
                                ('ens', VotingClassifier(estimators=estimators, voting='soft'))])

    ensemble.fit(X_train, y_train)
    models["Ensemble"] = ensemble
    
    
    return models

def test_models(models, Xy_test):
    X_test, y_test = Xy_test
    auc_list = []

    for name in models.keys():
        model = models[name]
        print("----------" + name + "----------")
        y_pred_prob = model.predict_proba(X_test)[::,1]
        auc = roc_auc_score(y_test, y_pred_prob)
        AUC_models[name].append(auc)
        auc_list.append(auc)
        print(auc)

    return auc_list
        
    




if __name__ == "__main__":
    df = load_data()
    df_feat = feature_extraction(get_mol(df))
    df_clean = data_cleaning(df_feat)
    # print("AUC score base")
    # test_models(modelling(transform(split(df_clean, 'train')), split(df_clean, 'train')), split(df_clean, 'test'))
    # print("AUC score oversampled")
    # test_models(modelling(transform(oversampling(split(df_clean, 'train'))), oversampling(split(df_clean, 'train'))), split(df_clean, 'test'))
    # print("AUC score undersampled")
    # test_models(modelling(transform(undersampling(split(df_clean, 'train'))), undersampling(split(df_clean, 'train'))), split(df_clean, 'test'))
    # print("AUC score syntetic samples")
    # test_models(modelling(transform(syntetic_samples(split(df_clean, 'train'))), syntetic_samples(split(df_clean, 'train'))), split(df_clean, 'test'))
    # print("AUC score syntetic undersampled")
    # test_models(modelling(transform(syntetic_under(split(df_clean, 'train'))), syntetic_under(split(df_clean, 'train'))), split(df_clean, 'test'))
    auc_base = []
    auc_over = []
    auc_under = []
    auc_synt = []
    auc_synt_under = []
    for i in range(50):
        print(i)
        auc_base_i = (test_models(modelling(transform(split(df_clean, 'train')), split(df_clean, 'train')), split(df_clean, 'test')))
        auc_over_i = (test_models(modelling(transform(oversampling(split(df_clean, 'train'))), oversampling(split(df_clean, 'train'))), split(df_clean, 'test')))
        auc_under_i = (test_models(modelling(transform(undersampling(split(df_clean, 'train'))), undersampling(split(df_clean, 'train'))), split(df_clean, 'test')))
        auc_synt_i = (test_models(modelling(transform(syntetic_samples(split(df_clean, 'train'))), syntetic_samples(split(df_clean, 'train'))), split(df_clean, 'test')))
        auc_synt_under_i = (test_models(modelling(transform(syntetic_under(split(df_clean, 'train'))), syntetic_under(split(df_clean, 'train'))), split(df_clean, 'test')))
        
        auc_base.extend(auc_base_i)
        auc_over.extend(auc_over_i)
        auc_under.extend(auc_under_i)
        auc_synt.extend(auc_synt_i)
        auc_synt_under.extend(auc_synt_under_i)
        


    print("----------------------|||||||||||-|||||||||||----------------------")
    print("AUC score base")
    print(np.mean(auc_base))
    print("AUC score oversampled")
    print(np.mean(auc_over))
    print("AUC score undersampled")
    print(np.mean(auc_under))
    print("AUC score syntetic samples")
    print(np.mean(auc_synt))
    print("AUC score syntetic undersampled")
    print(np.mean(auc_synt_under))

    print("----------------------|||||||||||-|||||||||||----------------------")
    for model in AUC_models.keys():
        print(model)
        print(np.mean(AUC_models[model]))
