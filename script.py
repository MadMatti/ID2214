import numpy as np
import pandas as pd
from IPython.display import display
from tabulate import tabulate
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

    
# Insert the functions create_column_filter and apply_column_filter below (after the comments)
#
# Input to create_column_filter:
# df - a dataframe (where the column names "CLASS" and "ID" have special meaning)
#
# Output from create_filter:
# df            - a new dataframe, where columns, except "CLASS" and "ID", containing only missing values 
#                 or only one unique value (apart from the missing values) have been dropped
# column_filter - a list of the names of the remaining columns, including "CLASS" and "ID"
#
# Hint 1: First copy the input dataframe and modify the copy (the input dataframe should be kept unchanged)
#
# Hint 2: Iterate through all columns and consider to drop a column only if it is not labeled "CLASS" or "ID"
#
# Hint 3: You may check the number of unique (non-missing) values in a column by applying the pandas functions
#         dropna and unique to drop missing values and get the unique (remaining) values
#
# Input to apply_column_filter:
# df            - a dataframe
# column_filter - a list of the names of the columns to keep (see above)
#
# Output from apply_column_filter:
# df - a new dataframe, where each column that is not included in column_filter has been dropped
#
# Hint 1: First copy the input dataframe and modify the copy (the input dataframe should be kept unchanged)

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

def check_result1a():
    df = pd.DataFrame({"CLASS":[1,0,1,0,1],"A":[1,2,np.nan,4,5],"B":[1,1,1,1,np.nan],"C":["h","h",np.nan,"i","h"],"D":[np.nan,np.nan,np.nan,np.nan,np.nan]})

    filtered_df, column_filter = create_column_filter(df)

    new_df = pd.DataFrame({"CLASS":[1,0,0],"A":[4,5,6],"B":[1,2,1],"C":[np.nan,np.nan,np.nan],"D":[np.nan,4,5]})

    filtered_new_df = apply_column_filter(new_df,column_filter)

    # display("df",df)
    # display("filtered_df",filtered_df)
    # display("new_df",new_df)
    # display("filtered_new_df",filtered_new_df)
    print("df")
    print(tabulate(df, headers='keys', tablefmt='psql'))
    print("filtered_df")
    print(tabulate(filtered_df, headers='keys', tablefmt='psql'))
    print("new_df")
    print(tabulate(new_df, headers='keys', tablefmt='psql'))
    print("filtered_new_df")
    print(tabulate(filtered_new_df, headers='keys', tablefmt='psql'))


# Insert the functions create_normalization and apply_normalization below (after the comments)
#
# Input to create_normalization:
# df: a dataframe (where the column names "CLASS" and "ID" have special meaning)
# normalizationtype: "minmax" (default) or "zscore"
#
# Output from create_normalization:
# df            - a new dataframe, where each numeric value in a column has been replaced by a normalized value
# normalization - a mapping (dictionary) from each column name to a triple, consisting of
#                ("minmax",min_value,max_value) or ("zscore",mean,std)
#
# Hint 1: First copy the input dataframe and modify the copy (the input dataframe should be kept unchanged)
#
# Hint 2: Consider columns of type "float" or "int" only (and which are not labeled "CLASS" or "ID"),
#         the other columns should remain unchanged
#
# Hint 3: Take a close look at the lecture slides on data preparation
#
# Input to apply_normalization:
# df            - a dataframe
# normalization - a mapping (dictionary) from column names to triples (see above)
#
# Output from apply_normalization:
# df - a new dataframe, where each numerical value has been normalized according to the mapping
#
# Hint 1: First copy the input dataframe and modify the copy (the input dataframe should be kept unchanged)
#
# Hint 2: For minmax-normalization, you may consider to limit the output range to [0,1]

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

def check_result1b():
    glass_train_df = pd.read_csv("glass_train.csv")
    glass_test_df = pd.read_csv("glass_test.csv")

    glass_train_norm, normalization = create_normalization(glass_train_df,normalizationtype="minmax")
    print("normalization:\n")
    for f in normalization:
        print("{}:{}".format(f,normalization[f]))

    print()
        
    glass_test_norm = apply_normalization(glass_test_df,normalization)
    display("glass_test_norm",glass_test_norm)


# Insert the functions create_imputation and apply_imputation below (after the comments)
#
# Input to create_imputation:
# df: a dataframe (where the column names "CLASS" and "ID" have special meaning)
#
# Output from create_imputation:
# df         - a new dataframe, where each missing numeric value in a column has been replaced by the mean of that column 
#              and each missing categoric value in a column has been replaced by the mode of that column
# imputation - a mapping (dictionary) from column name to value that has replaced missing values
#
# Hint 1: First copy the input dataframe and modify the copy (the input dataframe should be kept unchanged)
#
# Hint 2: Handle columns of type "float" or "int" only (and which are not labeled "CLASS" or "ID") in one way
#         and columns of type "object" and "category" in other ways
#
# Hint 3: Consider using the pandas functions mean and mode respectively, as well as fillna
#
# Hint 4: In the rare case of all values in a column being missing*, replace numeric values with 0,
#         object values with "" and category values with the first category (cat.categories[0])  
#
#         *Note that this will not occur if the previous column filter function has been applied
#
# Input to apply_imputation:
# df         - a dataframe
# imputation - a mapping (dictionary) from column name to value that should replace missing values
#
# Output from apply_imputation:
# df - a new dataframe, where each missing value has been replaced according to the mapping
#
# Hint 1: First copy the input dataframe and modify the copy (the input dataframe should be kept unchanged)
#
# Hint 2: Consider using fillna

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

def check_result1c():
    anneal_train_df = pd.read_csv("anneal_train.csv")
    anneal_test_df = pd.read_csv("anneal_test.csv")

    anneal_train_imp, imputation = create_imputation(anneal_train_df)
    anneal_test_imp = apply_imputation(anneal_test_df,imputation)

    print("Imputation:\n")
    for f in imputation:
        print("{}:{}".format(f,imputation[f]))

    print("\nNo. of replaced missing values in training data:\n{}".format(anneal_train_imp.count()-anneal_train_df.count()))
    print("\nNo. of replaced missing values in test data:\n{}".format(anneal_test_imp.count()-anneal_test_df.count()))


# Insert the functions create_bins and apply_bins below
#
# Input to create_bins:
# df      - a dataframe
# nobins  - no. of bins (default = 10)
# bintype - either "equal-width" (default) or "equal-size" 
#
# Output from create_bins:
# df      - a new dataframe, where each numeric feature value has been replaced by a categoric (corresponding to some bin)
# binning - a mapping (dictionary) from column name to bins (threshold values for the bin)
#
# Hint 1: First copy the input dataframe and modify the copy (the input dataframe should be kept unchanged)
#
# Hint 2: Discretize columns of type "float" or "int" only (and which are not labeled "CLASS" or "ID")
#
# Hint 3: Consider using pd.cut and pd.qcut respectively, with labels=False and retbins=True
#
# Hint 4: Set all columns in the new dataframe to be of type "category"
#
# Hint 5: Set the categories of the discretized features to be [0,...,nobins-1]
#
# Hint 6: Change the first and the last element of each binning to -np.inf and np.inf respectively 
#
# Input to apply_bins:
# df      - a dataframe
# binning - a mapping (dictionary) from column name to bins (threshold values for the bin)
#
# Output from apply_bins:
# df - a new dataframe, where each numeric feature value has been replaced by a categoric (corresponding to some bin)
#
# Hint 1: First copy the input dataframe and modify the copy (the input dataframe should be kept unchanged)
#
# Hint 2: Consider using pd.cut 
#
# Hint 3: Set all columns in the new dataframe to be of type "category"
#
# Hint 4: Set the categories of the discretized features to be [0,...,nobins-1]

def create_bins(df,nobins=10,bintype="equal-width"):
    df1 = df.copy()
    binning = {}
    for col in df.columns:
        if col not in ['CLASS', 'ID']:
            if df[col].dtype == 'float64' or df[col].dtype == 'int64':
                if bintype == 'equal-width':
                    df1[col], binning[col] = pd.cut(df[col], nobins, labels=False, retbins=True)
                elif bintype == 'equal-size':
                    df1[col], binning[col] = pd.qcut(df[col], nobins, labels=False, retbins=True, duplicates='drop')
                binning[col][0] = -np.inf
                binning[col][-1] = np.inf
                df1[col] = df1[col].astype('category')
                df1[col].cat.set_categories([i for i in range(nobins)], inplace=True)
    return df1, binning

def apply_bins(df,binning):
    df1 = df.copy()
    for col in df1.columns:
        if col in binning.keys():
            df1[col] = pd.cut(df1[col], binning[col], labels=False)
            df1[col] = df1[col].astype('category')
            df1[col] = df1[col].cat.set_categories([i for i in range(len(binning[col])-1)], inplace=True)
    return df1

def check_result1d():
    glass_train_df = pd.read_csv("glass_train.csv")

    glass_test_df = pd.read_csv("glass_test.csv")

    glass_train_disc, binning = create_bins(glass_train_df,nobins=10,bintype="equal-size")
    print("binning:")
    for f in binning:
        print("{}:{}".format(f,binning[f]))

    print()    
    glass_test_disc = apply_bins(glass_test_df,binning)
    display("glass_test_disc",glass_test_disc)


# Insert the functions create_one_hot and apply_one_hot below
#
# Input to create_one_hot:
# df: a dataframe
#
# Output from create_one_hot:
# df      - a new dataframe, where each categoric feature has been replaced by a set of binary features 
#           (as many new features as there are possible values)
# one_hot - a mapping (dictionary) from column name to a set of categories (possible values for the feature)
#
# Hint 1: First copy the input dataframe and modify the copy (the input dataframe should be kept unchanged)
#
# Hint 2: Consider columns of type "object" or "category" only (and which are not labeled "CLASS" or "ID")
#
# Hint 3: Consider creating new column names by merging the original column name and the categorical value
#
# Hint 4: Set all new columns to be of type "float"
#
# Hint 5: Do not forget to remove the original categoric feature
#
# Input to apply_one_hot:
# df      - a dataframe
# one_hot - a mapping (dictionary) from column name to categories
#
# Output from apply_one_hot:
# df - a new dataframe, where each categoric feature has been replaced by a set of binary features
#
# Hint: See the above Hints

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

def check_result1e():
    train_df = pd.read_csv("tic-tac-toe_train.csv")

    new_train, one_hot = create_one_hot(train_df)

    test_df = pd.read_csv("tic-tac-toe_test.csv")

    new_test_df = apply_one_hot(test_df,one_hot)
    display("new_test_df",new_test_df)


# Insert the function split below
#
# Input to split:
# df           - a dataframe
# testfraction - a float in the range (0,1) (default = 0.5)
#
# Output from split:
# trainingdf - a dataframe consisting of a random sample of (1-testfraction) of the rows in df
# testdf     - a dataframe consisting of the rows in df that are not included in trainingdf
#
# Hint: You may use np.random.permutation(df.index) to get a permuted list of indexes where a 
#       prefix corresponds to the test instances, and the suffix to the training instances 

def split(df,testfraction=0.5):
    testdf = df.loc[np.random.permutation(df.index)[:int(len(df)*testfraction)]]
    trainingdf = df.loc[~df.index.isin(testdf.index)]
    print(testdf.shape[0], trainingdf.shape[0])
    return trainingdf, testdf

def split2(df,testfraction=0.5):
    testdf = df.sample(frac=testfraction)
    trainingdf = df.drop(testdf.index)
    print(testdf.shape[0], trainingdf.shape[0])
    return trainingdf, testdf

def check_result1f():
    glass_df = pd.read_csv("glass.csv")

    glass_train, glass_test = split2(glass_df,testfraction=0.25)

    print("Training IDs:\n{}".format(glass_train["ID"].values))

    print("\nTest IDs:\n{}".format(glass_test["ID"].values))

    print("\nOverlap: {}".format(set(glass_train["ID"]).intersection(set(glass_test["ID"]))))


# Insert the function accuracy below
#
# Input to accuracy:
# df            - a dataframe with class labels as column names and each row corresponding to
#                 a prediction with estimated probabilities for each class
# correctlabels - an array (or list) of the correct class label for each prediction
#                 (the number of correct labels must equal the number of rows in df)
#
# Output from accuracy:
# accuracy - the fraction of cases for which the predicted class label coincides with the correct label
#
# Hint: In case the label receiving the highest probability is not unique, you may
#       resolve that by picking the first (as ordered by the column names) or 
#       by randomly selecting one of the labels with highest probaility.

def accuracy(df,correctlabels):
    correctpred = 0
    for i in range(len(correctlabels)):
        if df.iloc[i].idxmax() == correctlabels[i]: # idxmax() returns the index of the first max value
            correctpred += 1
    accuracy = correctpred / len(correctlabels)
    return accuracy

def check_result1g():
    predictions = pd.DataFrame({"A":[0.5,0.5,0.5,0.25,0.25],"B":[0.5,0.25,0.25,0.5,0.25],"C":[0.0,0.25,0.25,0.25,0.5]})
    display("predictions",predictions)

    correctlabels = ["B","A","B","B","C"]

    print("Accuracy: {}".format(accuracy(predictions,correctlabels)))


# Insert the function folds below
#
# Input to folds:
# df      - a dataframe
# nofolds - an integer greater than 1 (default = 10)
#
# Output from folds:
# folds - a list (of length = nofolds) dataframes consisting of random non-overlapping, 
#         approximately equal-sized subsets of the rows in df
#
# Hint: You may use np.random.permutation(df.index) to get a permuted list of indexes from which a 
#       prefix corresponds to the test instances, and the suffix to the training instances 


def folds(df,nofolds=10):
    shuffle = df.sample(frac=1)
    folds = np.array_split(shuffle, nofolds)
    return folds

def check_result2a():
    glass_df = pd.read_csv("glass.csv")

    glass_folds = folds(glass_df,nofolds=5)

    fold_sizes = [len(f) for f in glass_folds]

    print("Fold sizes:{}\nTotal no. instances: {}".format(fold_sizes,sum(fold_sizes))) 


# Insert the function brier_score below
#
# Input to brier_score:
# df            - a dataframe with class labels as column names and each row corresponding to
#                 a prediction with estimated probabilities for each class
# correctlabels - an array (or list) of the correct class label for each prediction
#                 (the number of correct labels must equal the number of rows in df)
#
# Output from brier_score:
# brier_score - the average square error of the predicted probabilties 
#
# Hint: Compare each predicted vector to a vector for each correct label, which is all zeros except 
#       for at the index of the correct class. The index can be found using np.where(df.columns==l)[0] 
#       where l is the correct label.

def brier_score(df,correctlabels):
    brier_score = 0
    for i in range(len(correctlabels)):
        correctvec = np.zeros(len(df.columns))
        correctvec[np.where(df.columns==correctlabels[i])[0]] = 1
        brier_score += np.sum((df.iloc[i] - correctvec)**2)
    brier_score = brier_score / len(correctlabels)
    return brier_score

def check_result2b():
    predictions = pd.DataFrame({"A":[0.5,0.5,0.5,0.25,0.25],"B":[0.5,0.25,0.25,0.5,0.25],"C":[0.0,0.25,0.25,0.25,0.5]})

    correctlabels = ["B","A","B","B","C"]

    print("Brier score: {}".format(brier_score(predictions,correctlabels)))


# Insert the function auc below
#
# Input to auc:
# df            - a dataframe with class labels as column names and each row corresponding to
#                 a prediction with estimated probabilities for each class
# correctlabels - an array (or list) of the correct class label for each prediction
#                 (the number of correct labels must equal the number of rows in df)
#
# Output from auc:
# auc - the weighted area under ROC curve
#
# Hint 1: Calculate the binary AUC first for each class label c, i.e., treating the
#         predicted probability of this class for each instance as a score; the true positives
#         are the ones belonging to class c and the false positives the rest
#
# Hint 2: When calculating the binary AUC, first find the scores of the true positives and then
#         the scores of the true negatives
#
# Hint 3: You may use a dictionary with a mapping from each score to an array of two numbers; 
#         the number of true positives with this score and the number of true negatives with this score
#
# Hint 4: Created a (reversely) sorted (on the scores) list of pairs from the dictionary and
#         iterate over this to additively calculate the AUC
#
# Hint 5: For each pair in the above list, there are three cases to consider; the no. of false positives
#         is zero, the no. of true positives is zero, and both are non-zero
#
# Hint 6: Calculate the weighted AUC by summing the individual AUCs weighted by the relative
#         frequency of each class (as estimated from the correct labels)


def auc_binary(df, correctlabels, c):
    df1 = df.copy()
    df1.drop(df1.columns.difference([c]), axis=1, inplace=True)
    df1["actual"] = correctlabels
    df1.rename({c: 'prediction'}, axis=1, inplace=True)
    columns_titles = ["actual","prediction"]
    df1 = df1.reindex(columns=columns_titles)

    thresholds = list(np.array(list(range(0,1001,1)))/100)
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
    pivot["threshold"] = thresholds
    # plt.scatter(pivot.y, pivot.x)
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.show()

    auc = abs(np.trapz(pivot.x, pivot.y))

    return auc

def auc(df,correctlabels):
    auc = 0
    for c in df.columns:
        auc += auc_binary(df, correctlabels, c) / len(correctlabels) * correctlabels.count(c)
    return auc

def check_result2c():
    predictions = pd.DataFrame({"A":[0.9,0.9,0.6,0.55],"B":[0.1,0.1,0.4,0.45]})

    correctlabels = ["A","B","B","A"]

    print("AUC: {}".format(auc(predictions,correctlabels)))

def check_result2c_1():
    predictions = pd.DataFrame({"A":[0.5,0.5,0.5,0.25,0.25],"B":[0.5,0.25,0.25,0.5,0.25],"C":[0.0,0.25,0.25,0.25,0.5]})

    correctlabels = ["B","A","B","B","C"]

    print("AUC: {}".format(auc(predictions,correctlabels)))



                
                
        


if __name__ == "__main__":
    # check_result1a()
    # check_result1b()
    # check_result1c()
    # check_result1d()
    #check_result1e()
    #check_result1f()
    #check_result1g()
    #check_result2a()
    #check_result2b()
    check_result2c()
    check_result2c_1()
