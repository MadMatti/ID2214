import pandas as pd
import numpy as np
import sklearn
import time
from sklearn.tree import DecisionTreeClassifier



# function from Assignment 1

# column filter
def create_column_filter(df):
    df1 = df.copy()
    column_filter = ['CLASS']
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

def auc_binary(predictions, correctlabels, threshold, c):
    # array with true for correct labels for class c (by row index)
    correctlabels_class = np.array(correctlabels)==predictions.columns[c]
    # array with predictions for all instances that should be classified class c
    predictions_class = predictions[predictions.columns[c]]
    # array with true for all correctly predicted labels according to threshold
    predicted_labels = predictions_class[correctlabels_class] >= threshold
    pos = sum(predicted_labels) # number of correctly predicted labels
    # tp / (tp + fn)
    tpr = pos / sum(correctlabels_class)

    # same reasoning for negative class
    not_correctlabels_class = np.array(correctlabels)!=predictions.columns[c]
    predictions_class = predictions[ predictions.columns[c] ]
    predicted_labels = predictions_class[not_correctlabels_class] >= threshold
    neg = sum(predicted_labels)
    # fpr = fp / (fp + tn)
    fpr = neg / sum(not_correctlabels_class)
    
    return tpr, fpr


def auc(predictions, correctlabels):
    thresholds = np.unique(predictions)
    AUC_d = {}
    
    # iterate over all classes and calculate the area under the ROC(tpr/fpr) curve (AUC)
    for (index,c) in enumerate(np.unique(correctlabels)):
        roc_points = [auc_binary(predictions, correctlabels, th, index) for th in reversed(thresholds)]
                    
        # calculate AUC as area under the curve
        AUC = 0
        tpr_last = 0
        fpr_last = 0
        
        # iterate over all thresholds
        for r in roc_points:
            tpr, fpr = r
            # Add area under triangle        
            if tpr > tpr_last and fpr > fpr_last:
                AUC += (fpr-fpr_last)*tpr_last + (fpr-fpr_last)*(tpr-tpr_last) / 2
            # Add area under rectangle            
            elif fpr > fpr_last:
                AUC += (fpr-fpr_last)*tpr
            # update point coordinates (tpr, fpr) of curve
            tpr_last = tpr
            fpr_last = fpr
       
        AUC_d[c] = AUC
        
    # take the weighted average for all classes
    AUC_total = 0
    for (cName,auc) in AUC_d.items():
        number_of_labels = np.sum(np.array(correctlabels) == cName)
        weight = number_of_labels / len(correctlabels)
        AUC_total += weight * auc
        
    return AUC_total


# Define the class RandomForest with three functions __init__, fit and predict (after the comments):
#
# Input to __init__: 
# self - the object itself
#
# Output from __init__:
# <nothing>
# 
# This function does not return anything but just initializes the following attributes of the object (self) to None:
# column_filter, imputation, one_hot, labels, model
#
# Input to fit:
# self      - the object itself
# df        - a dataframe (where the column names "CLASS" and "ID" have special meaning)
# no_trees  - no. of trees in the random forest (default = 100)
#
# Output from fit:
# <nothing>
#
# The result of applying this function should be:
#
# self.column_filter - a column filter (see Assignment 1) from df
# self.imputation    - an imputation mapping (see Assignment 1) from df
# self.one_hot       - a one-hot mapping (see Assignment 1) from df
# self.labels        - a (sorted) list of the categories of the "CLASS" column of df
# self.model         - a random forest, consisting of no_trees trees, where each tree is generated from a bootstrap sample
#                      and the number of evaluated features is log2|F| where |F| is the total number of features
#                      (for details, see lecture slides)
#
# Note that the function does not return anything but just assigns values to the attributes of the object.
#
# Hint 1: First create the column filter, imputation and one-hot mappings
#
# Hint 2: Then get the class labels and the numerical values (as an ndarray) from the dataframe after dropping the class labels 
#
# Hint 3: Generate no_trees classification trees, where each tree is generated using DecisionTreeClassifier 
#         from a bootstrap sample (see lecture slides), e.g., generated by np.random.choice (with replacement) 
#         from the row numbers of the ndarray, and where a random sample of the features are evaluated in
#         each node of each tree, of size log2(|F|), where |F| is the total number of features;
#         see the parameter max_features of DecisionTreeClassifier
#
# Input to predict:
# self - the object itself
# df   - a dataframe
# 
# Output from predict:
# predictions - a dataframe with class labels as column names and the rows corresponding to
#               predictions with estimated class probabilities for each row in df, where the class probabilities
#               are the averaged probabilities output by each decision tree in the forest
#
# Hint 1: Drop any "CLASS" and "ID" columns of the dataframe first and then apply column filter, imputation and one_hot
#
# Hint 2: Iterate over the trees in the forest to get the prediction of each tree by the method predict_proba(X) where 
#         X are the (numerical) values of the transformed dataframe; you may get the average predictions of all trees,
#         by first creating a zero-matrix with one row for each test instance and one column for each class label, 
#         to which you add the prediction of each tree on each iteration, and then finally divide the prediction matrix
#         by the number of trees.
#
# Hint 3: You may assume that each bootstrap sample that was used to generate each tree has included all possible
#         class labels and hence the prediction of each tree will contain probabilities for all class labels
#         (in the same order). Note that this assumption may be violated, and this limitation will be addressed 
#         in the next part of the assignment. 

class RandomForest:

    def __init__(self):
        self.column_filter = None
        self.imputation = None
        self.one_hot = None
        self.labels = None
        self.model = None
        self.training_labels = None

    def fit(self, dt, no_trees=100):
        df1 = dt.copy()
        df1, self.column_filter  = create_column_filter(df1)
        df1, self.imputation = create_imputation(df1)
        df1, self.one_hot = create_one_hot(df1)
        self.labels = list(df1["CLASS"].astype("category").cat.categories)
        features = df1.drop(columns=["ID", "CLASS"], errors='ignore').to_numpy()
        
        models = []
                
        for i in range(no_trees):
            row_nums=np.random.choice(len(features),len(features), replace=True)
            data = features[row_nums]
            labels=[df1['CLASS'].astype("category")[i] for i in row_nums]
            tree = DecisionTreeClassifier(max_features='log2')
            tree.fit(data,labels)
            models.append(tree)
        
        self.model=models

    def predict(self, df):
        df1 = df.copy()
        df1 = apply_column_filter(df1, self.column_filter)
        df1 = apply_imputation(df1, self.imputation)
        df1 = apply_one_hot(df1,self.one_hot)
        df1.drop(columns=["CLASS"], inplace=True)
                
        probabilities = np.zeros((len(df1),len(self.labels)))
        
        for tree in self.model:
            for i, row in enumerate(df1.values):
                result = tree.predict_proba(row.reshape(1,-1))
                probabilities[i] = probabilities[i] + result

        probabilities = probabilities / len(self.model)
        predictions = pd.DataFrame(probabilities, columns=self.labels)

        return predictions


# check results
def check_1(): 
    train_df = pd.read_csv("tic-tac-toe_train.csv")

    test_df = pd.read_csv("tic-tac-toe_test.csv")

    rf = RandomForest()

    t0 = time.perf_counter()
    rf.fit(train_df)
    print("Training time: {:.2f} s.".format(time.perf_counter()-t0))

    test_labels = test_df["CLASS"]

    t0 = time.perf_counter()
    predictions = rf.predict(test_df)

    print("Testing time: {:.2f} s.".format(time.perf_counter()-t0))

    print("Accuracy: {:.4f}".format(accuracy(predictions,test_labels)))
    print("AUC: {:.4f}".format(auc(predictions,test_labels))) # Comment this out if not implemented in assignment 1
    print("Brier score: {:.4f}".format(brier_score(predictions,test_labels))) # Comment this out if not implemented in assignment 1

    train_labels = train_df["CLASS"]
    predictions = rf.predict(train_df)
    print("Accuracy on training set: {0:.4f}".format(accuracy(predictions,train_labels)))
    print("AUC on training set: {0:.4f}".format(auc(predictions,train_labels))) # Comment this out if not implemented in assignment 1
    print("Brier score on training set: {0:.4f}".format(brier_score(predictions,train_labels))) # Comment this out if not implemented in assignment 1









if __name__ == "__main__":
    check_1()





