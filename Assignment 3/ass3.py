import numpy as np
import pandas as pd
import time
import sklearn
from sklearn.tree import DecisionTreeClassifier

# function from assignment 1

# column filter
def create_column_filter(df):
    df1 = df.copy()
    column_filter = ['CLASS', 'ID']
    for col in df.columns:
        if col not in ['CLASS', 'ID']:
            if len(df[col].dropna().unique()) > 1:
                column_filter.append(col)
    df1 = df1[column_filter]
    return df1, column_filter

def apply_column_filter(df, column_filter):
    df1 = df.copy()
    df1 = df[column_filter]
    return df1

# normalization
def create_normalization(df,normalizationtype="minmax"):
    df1 = df.copy()
    normalization = {}
    for col in df.columns:
        if col not in ['CLASS', 'ID']:
            if df[col].dtype == 'float64' or df[col].dtype == 'int64':
                if normalizationtype == "minmax":
                    normalization[col] = ("minmax", df[col].min(), df[col].max())
                    df1[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
                elif normalizationtype == "zscore":
                    normalization[col] = ("zscore", df[col].mean(), df[col].std())
                    df1[col] = (df[col] - df[col].mean()) / df[col].std()
    return df1, normalization

def apply_normalization(df,normalization):
    df1 = df.copy()
    for col in df.columns:
        if col in normalization.keys():
            if normalization[col][0] == "minmax":
                df1[col] = (df[col] - normalization[col][1]) / (normalization[col][2] - normalization[col][1])
            elif normalization[col][0] == "zscore":
                df1[col] = (df[col] - normalization[col][1]) / normalization[col][2]
    return df1

# imputation
def create_imputation(df):
    df1 = df.copy()
    imputation = {}
    for col in df.columns:
        if col not in ['CLASS', 'ID']:
            if df[col].dtype == 'float64' or df[col].dtype == 'int64':
                imputation[col] = df[col].mean()
                df1[col] = df[col].fillna(imputation[col])
            elif df[col].dtype == 'object' or df[col].dtype == 'category':
                imputation[col] = df[col].mode()[0]
                df1[col] = df[col].fillna(imputation[col])
    return df1, imputation

def apply_imputation(df,imputation):
    df1 = df.copy()
    for col in df.columns:
        if col in imputation.keys():
            df1[col] = df[col].fillna(imputation[col])
    return df1

# binning
def create_bins(df, nobins=10, bintype="equal-width"):
    df1 = df.copy()
    binning = {}

    for col in df1.columns:
        if col not in ['CLASS', 'ID']:
            if df1[col].dtype == "int64" or df1[col].dtype == "float64":
                if bintype == "equal-width":
                    df1[col], binning[col] = pd.cut(df1[col],nobins,labels=False, retbins=True)
                elif bintype == "equal-size":
                    df1[col], binning[col] = pd.qcut(df[col],nobins, labels=False, retbins=True, duplicates='drop')

                binning[col][0] = -np.inf
                binning[col][-1] = np.inf
                df1[col] = df1[col].astype('category')
                df1[col] = df1[col].cat.set_categories([str(i) for i in df1[col].cat.categories], rename = True)

    return df1, binning

def apply_bins(df, binning):
    df1 = df.copy()

    for col in df1:
        if col in binning.keys():
            df1[col] = pd.cut(df1[col], binning[col], labels=False)
            df1[col] = df1[col].astype('category')
            df1[col] = df1[col].cat.set_categories([str(i) for i in df1[col].cat.categories], rename = True)

    return df1

# one-hot encoding
def create_one_hot(df):
    df1 = df.copy()
    one_hot = {}
    for col in df.columns:
        if col not in ['CLASS', 'ID']:
            if df[col].dtype == 'object' or df[col].dtype == 'category':
                one_hot[col] = df[col].unique()
                for val in one_hot[col]:
                    df1[col + '_' + str(val)] = (df1[col] == val).astype('float')
                df1.drop(col, axis=1, inplace=True)
    return df1, one_hot

def apply_one_hot(df,one_hot):
    df1 = df.copy()
    for col in one_hot.keys():
        for val in one_hot[col]:
            df1[col + '_' + str(val)] = (df1[col] == val).astype('float')
        df1.drop(col, axis=1, inplace=True)
    return df1

# split
def split(df, testfraction=0.5):
    randind = np.random.permutation(df.index)
    testind = randind[:int(len(randind)*testfraction)]
    testdf = df.iloc[testind]
    traindf = df.drop(testind)
    return traindf, testdf

# accuracy prediction
def accuracy(df, correctlabels):

    correct_pred = 0

    for index, row in df.iterrows():
        maxind = row.argmax()
        predlabel = df.columns[maxind]
        if predlabel == correctlabels[index]:
            correct_pred += 1

    accuracy = correct_pred / len(correctlabels)
    return accuracy

# folds 
def folds(df,nofolds=10):
    shuffle = df.sample(frac=1)
    folds = np.array_split(shuffle, nofolds)
    return folds

# brier score
def brier_score(df, correctlabels):
    brier_score = 0
    n = len(df.index)

    for i,p in df.iterrows():
        true_label = correctlabels[i]
        o = np.zeros(len(p))
        o[df.columns==true_label] = 1
        brier_score += ((p-o)**2).sum(axis=0)
    brier_score = brier_score/n
        
    return brier_score

# auc
def auc_binary(df, correctlabels, c):
    # Only some modification to the dataframe for better understanding
    df1 = df.copy()
    df1.drop(df1.columns.difference([c]), axis=1, inplace=True)
    df1["actual"] = correctlabels
    df1.rename({c: 'prediction'}, axis=1, inplace=True)
    columns_titles = ["actual","prediction"]
    df1 = df1.reindex(columns=columns_titles)

    thresholds = list(np.array(list(range(0,101,1)))/100)
    roc_points = []

    for threshold in thresholds:

        tp = tn = fp = fn = 0

        for index, instance in df1.iterrows():
            actual = instance["actual"] == c
            prediction = instance["prediction"]

            if prediction >= threshold:
                prediction_class = True
            else:
                prediction_class = False

            if prediction_class and actual:
                tp += 1
            elif prediction_class and not actual:
                fp += 1
            elif not prediction_class and actual:
                fn += 1
            elif not prediction_class and not actual:
                tn += 1

        tpr = tp / (tp + fn)
        fpr = fp / (fp + tn)
        roc_points.append([tpr, fpr])

    pivot = pd.DataFrame(roc_points, columns=['x', 'y'])

    auc = abs(np.trapz(pivot.x, pivot.y))

    return auc

def auc(df,correctlabels):
    auc = 0
    for c in df.columns:
        auc += auc_binary(df, correctlabels, c) / len(correctlabels) * list(correctlabels).count(c)
    return auc


#######

class RandomForest:
    
    def __init__(self):
        self.column_filter = None
        self.imputation = None
        self.one_hot = None
        self.labels = None
        self.model = None
        
    def fit(self, df, no_trees=100):
        df1 = df.copy()
        
        df1, self.column_filter = create_column_filter(df1)
        df1, self.imputation = create_imputation(df1)
        df1, self.one_hot = create_one_hot(df1)
        self.labels = df1['CLASS'].astype('category').cat.categories.tolist()
        
        trees = 0
        self.model = trees
    
        
