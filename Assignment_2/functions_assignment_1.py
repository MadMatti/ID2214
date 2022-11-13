import numpy as np
import pandas as pd
from platform import python_version

print(f"Python version: {python_version()}")
print(f"NumPy version: {np.__version__}")
print(f"Pandas version: {pd.__version__}")

def create_column_filter(df):
    
    df1 = df.copy()
    df1.dropna(how='all', axis=1, inplace=True) # drop columns full of NaN's
    column_filter = ['CLASS']

    for column in df1:
        values = df1[column]
        unique_values = np.unique(values[~pd.isnull(values)])
        if column not in ['CLASS', 'ID']: 
            if len(unique_values) < 2:
                df1.drop(column, axis=1, inplace=True)
    column_filter = list(df1.columns.values)
    
    return df1, column_filter

def apply_column_filter(df, column_filter):
    df1 = df.copy()
    for column in df1:
        if column not in column_filter:
            df1.drop(column, axis=1, inplace=True)
    return df1    

def create_normalization(df, normalizationtype="minmax"):
    df1 = df.copy()
    normalization = {}

    for col in df1:
        if col not in ['CLASS', 'ID']:
            if df1[col].dtype == "int64" or df1[col].dtype == "float64":
                if normalizationtype == "minmax":
                    minv = df1[col].min()
                    maxv = df1[col].max()
                    df1[col] = [(x-minv)/(maxv-minv) for x in df1[col]]
                    normalization[col] = ("minmax", minv, maxv)
                if normalizationtype == "zscore":
                    mean = df1[col].mean()
                    std = df1[col].std()
                    df1[col] = df1[col].apply(lambda x: (x-mean)/std)
                    normalization[col] = ("zscore", mean, std)
    return df1, normalization

def apply_normalization(df, normalization):
    df1 = df.copy()

    for col in df1:
        if col in normalization.keys() and col not in ['CLASS', 'ID']:
            if normalization[col][0] == "minmax":
                minv = normalization[col][1]
                maxv = normalization[col][2]
                df1[col] = [(x-minv)/(maxv-minv) for x in df1[col]]
                df1[col].clip(0,1, inplace=True)
                
            if normalization[col][0] == "zscore":
                mean = normalization[col][1]
                std = normalization[col][2]
                df1[col] = df1[col] = df1[col].apply(lambda x: (x-mean)/std)
    return df1

def create_imputation(df):
    df1 = df.copy()
    imputation = {}

    for col in df1:
        if col not in ['CLASS', 'ID']:
            if df1[col].dtype == "int64" or df1[col].dtype == "float64":
                if df1[col].isnull().all():
                    df1[col].fillna(0, inplace=True)
                df1[col].fillna(df1[col].mean(),inplace=True)
                imputation[col] = df1[col].mean()
            elif df1[col].dtype == "object" or df1[col] == "category":
                if df1[col].isnull().all() and df1[col].dtype == "object":
                    df1[col].fillna("", inplace=True)
                elif df1[col].isnull().all() and df1[col].dtype == "object":
                    df1[col].fillna(df1[col].categories[0], inplace=True)

                df1[col].fillna(df1[col].mode()[0],inplace=True)
                imputation[col] = df1[col].mode()[0]
    return df1, imputation

def apply_imputation(df, imputation):
    df1 = df.copy()

    for col in df1:
        if col in imputation.keys():
            df1[col] = imputation[col]
    return df1

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

def create_one_hot(df):
    df1 = df.copy()
    one_hot = {}

    for col in df1.columns:
        if col not in ['CLASS', 'ID'] and (df1[col].dtype == 'category' or df1[col].dtype == 'object'):
            one_hot[col] = df[col].unique()
            for val in one_hot[col]:
                df1[col + '_' + str(val)] = (df1[col] == val).astype('float')
            df1.drop(col, axis=1, inplace=True)
    return df1, one_hot

def apply_one_hot(df, one_hot):
    df1 = df.copy()

    for col in df1.columns:
        if col in one_hot.keys():
            for val in one_hot[col]:
                df1[col + '-' + str(val)] = (df1[col] == val).astype('float')
            df1.drop(col, axis=1, inplace=True)
    return df1

def split(df, testfraction=0.5):
    randind = np.random.permutation(df.index)
    testind = randind[:int(len(randind)*testfraction)]
    testdf = df.iloc[testind]
    traindf = df.drop(testind)
    return traindf, testdf


def accuracy(df, correctlabels):

    correct_pred = 0

    for index, row in df.iterrows():
        maxind = row.argmax()
        predlabel = df.columns[maxind]
        if predlabel == correctlabels[index]:
            correct_pred += 1

    accuracy = correct_pred / len(correctlabels)
    return accuracy

