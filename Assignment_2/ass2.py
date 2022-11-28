import numpy as np
import pandas as pd
import time
from IPython.display import display

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
# def auc_binary(df, correctlabels, c):
#     # Only some modification to the dataframe for better understanding
#     df1 = df.copy()
#     df1.drop(df1.columns.difference([c]), axis=1, inplace=True)
#     df1["actual"] = correctlabels
#     df1.rename({c: 'prediction'}, axis=1, inplace=True)
#     columns_titles = ["actual","prediction"]
#     df1 = df1.reindex(columns=columns_titles)

#     thresholds = list(np.array(list(range(0,101,1)))/100)
#     roc_points = []

#     for threshold in thresholds:

#         tp = tn = fp = fn = 0

#         for index, instance in df1.iterrows():
#             actual = instance["actual"] == c
#             prediction = instance["prediction"]

#             if prediction >= threshold:
#                 prediction_class = True
#             else:
#                 prediction_class = False

#             if prediction_class and actual:
#                 tp += 1
#             elif prediction_class and not actual:
#                 fp += 1
#             elif not prediction_class and actual:
#                 fn += 1
#             elif not prediction_class and not actual:
#                 tn += 1

#         tpr = tp / (tp + fn)
#         fpr = fp / (fp + tn)

#         roc_points.append([tpr, fpr])

#     pivot = pd.DataFrame(roc_points, columns=['x', 'y'])

#     auc = abs(np.trapz(pivot.x, pivot.y))

#     return auc

# def auc(df,correctlabels):
#     auc = 0
#     for c in df.columns:
#         auc += auc_binary(df, correctlabels, c) / len(correctlabels) * list(correctlabels).count(c)
#     return auc

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


# Define the class kNN with three functions __init__, fit and predict (after the comments):
#
# Input to __init__: 
# self - the object itself
#
# Output from __init__:
# <nothing>
# 
# This function does not return anything but just initializes the following attributes of the object (self) to None:
# column_filter, imputation, normalization, one_hot, labels, training_labels, training_data, training_time
#
# Input to fit:
# self              - the object itself
# df                - a dataframe (where the column names "CLASS" and "ID" have special meaning)
# normalizationtype - "minmax" (default) or "zscore"
#
# Output from fit:
# <nothing>
#
# The result of applying this function should be:
#
# self.column_filter   - a column filter (see Assignment 1) from df
# self.imputation      - an imputation mapping (see Assignment 1) from df
# self.normalization   - a normalization mapping (see Assignment 1), using normalizationtype from the imputed df
# self.one_hot         - a one-hot mapping (see Assignment 1)
# self.training_labels - a pandas series corresponding to the "CLASS" column, set to be of type "category" 
# self.labels          - a list of the categories (class labels) of the previous series
# self.training_data   - the values (an ndarray) of the transformed dataframe, i.e., after employing imputation, 
#                        normalization, and possibly one-hot encoding, and also after removing the "CLASS" and "ID" columns
#
# Note that the function does not return anything but just assigns values to the attributes of the object.
#
# Input to predict:
# self - the object itself
# df   - a dataframe
# k    - an integer >= 1 (default = 5)
# 
# Output from predict:
# predictions - a dataframe with class labels as column names and the rows corresponding to
#               predictions with estimated class probabilities for each row in df, where the class probabilities
#               are estimated by the relative class frequencies in the set of class labels from the k nearest 
#               (with respect to Euclidean distance) neighbors in training_data
#
# Hint 1: Drop any "CLASS" and "ID" columns first and then apply column filtering, imputation, normalization and one-hot
#
# Hint 2: Get the numerical values (as an ndarray) from the resulting dataframe and iterate over the rows 
#         calling some sub-function, e.g., get_nearest_neighbor_predictions(x_test,k), which for a test row
#         (numerical input feature values) finds the k nearest neighbors and calculate the class probabilities.
#
# Hint 3: This sub-function may first find the distances to all training instances, e.g., pairs consisting of
#         training instance index and distance, and then sort them according to distance, and then (using the indexes
#         of the k closest instances) find the corresponding labels and calculate the relative class frequencies

class kNN:

    def __init__(self):
        self.column_filter = None
        self.imputation = None
        self.normalization = None
        self.one_hot = None
        self.labels = None
        self.training_labels = None
        self.training_data = None
        self.training_time = None

    def fit(self, df, normalizationtype='minmax'):
        df1 = df.copy()
        df1, self.column_filter = create_column_filter(df1)
        df1, self.imputation = create_imputation(df1)
        df1, self.normalization = create_normalization(df1, normalizationtype)
        df1, self.one_hot = create_one_hot(df1)
        self.training_labels = df1['CLASS'].astype('category')
        self.labels = list(self.training_labels.cat.categories)
        self.training_data = df1[df1.columns.difference(['CLASS', 'ID'])].to_numpy()

    def euclidean_distance(self, point1, point2):
        # calculate the euclidean distance between two points
        return np.linalg.norm(point1 - point2)

    def get_prediction(self, x_test, k):
        distances = np.empty(self.training_data.shape[0])

        for index, row in enumerate(self.training_data):
            distances[index] = self.euclidean_distance(row, x_test)

        # sort the array and get the sorted indeces
        sorted_indicies = distances.argsort()
        # take the first k rows (closest neighbors)
        k_indicies = sorted_indicies[:k]
        # get the labels of the closest neighbors
        k_labels = self.training_labels[k_indicies]
        # calculate the relative class frequencies
        unique, counts = np.unique(k_labels, return_counts=True)
        # calculate the probabilities
        probabilities = dict(zip(unique, counts/k))


        return probabilities

    def predict(self, df, k=5):
        df1 = df.copy()

        df1 = apply_column_filter(df1, self.column_filter)
        df1 = apply_imputation(df1, self.imputation)
        df1 = apply_normalization(df1, self.normalization)
        df1 = apply_one_hot(df1, self.one_hot)
        labels = np.unique(self.training_labels)
        columns = df1[df.columns.difference(['CLASS', 'ID'])]

        # get numerical values as ndarray
        values = (columns.select_dtypes(include=np.number)).to_numpy()
        # result dataframe
        result = pd.DataFrame(0.0, index=range(0, len(self.training_data)), columns=labels)

        for index, row in enumerate(values):
            # get the class probabilities
            probabilities = self.get_prediction(row, k)
            for prob in probabilities:
                # set the probability value in the result dataframe
                result.at[index, prob] = probabilities[prob]

        return result


# test code
def check_result_KNN():

    glass_train_df = pd.read_csv("glass_train.csv")

    glass_test_df = pd.read_csv("glass_test.csv")

    knn_model = kNN()

    t0 = time.perf_counter()
    knn_model.fit(glass_train_df)
    print("Training time: {0:.2f} s.".format(time.perf_counter()-t0))

    test_labels = glass_test_df["CLASS"]

    k_values = [1,3,5,7,9]
    results = np.empty((len(k_values),3))

    for i in range(len(k_values)):
        t0 = time.perf_counter()
        predictions = knn_model.predict(glass_test_df,k=k_values[i])
        print("Testing time (k={0}): {1:.2f} s.".format(k_values[i],time.perf_counter()-t0))
        results[i] = [accuracy(predictions,test_labels),brier_score(predictions,test_labels),
                    auc(predictions,test_labels)] # Assuming that you have defined auc - remove otherwise

    results = pd.DataFrame(results,index=k_values,columns=["Accuracy","Brier score","AUC"])

    print()
    display("results",results)


    train_labels = glass_train_df["CLASS"]
    predictions = knn_model.predict(glass_train_df,k=1)
    print("Accuracy on training set (k=1): {0:.4f}".format(accuracy(predictions,train_labels)))
    print("AUC on training set (k=1): {0:.4f}".format(auc(predictions,train_labels)))
    print("Brier score on training set (k=1): {0:.4f}".format(brier_score(predictions,train_labels)))



# Define the class NaiveBayes with three functions __init__, fit and predict (after the comments):
#
# Input to __init__: 
# self - the object itself
#
# Output from __init__:
# <nothing>
# 
# This function does not return anything but just initializes the following attributes of the object (self) to None:
# column_filter, binning, labels, class_priors, feature_class_value_counts, feature_class_counts
#
# Input to fit:
# self    - the object itself
# df      - a dataframe (where the column names "CLASS" and "ID" have special meaning)
# nobins  - no. of bins (default = 10)
# bintype - either "equal-width" (default) or "equal-size" 
#
# Output from fit:
# <nothing>
#
# The result of applying this function should be:
#
# self.column_filter              - a column filter (see Assignment 1) from df
# self.binning                    - a discretization mapping (see Assignment 1) from df
# self.class_priors               - a mapping (dictionary) from the labels (categories) of the "CLASS" column of df,
#                                   to the relative frequencies of the labels
# self.labels                     - a list of the categories (class labels) of the "CLASS" column of df
# self.feature_class_value_counts - a mapping from the feature (column name) to the number of
#                                   training instances with a specific combination of (non-missing, categorical) 
#                                   value for the feature and class label
# self.feature_class_counts       - a mapping from the feature (column name) to the number of
#                                   training instances with a specific class label and some (non-missing, categorical) 
#                                   value for the feature
#
# Note that the function does not return anything but just assigns values to the attributes of the object.
#
# Input to predict:
# self - the object itself
# df   - a dataframe
# 
# Output from predict:
# predictions - a dataframe with class labels as column names and the rows corresponding to
#               predictions with estimated class probabilities for each row in df, where the class probabilities
#               are estimated by the naive approximation of Bayes rule (see lecture slides)
#
# Hint 1: First apply the column filter and discretization
#
# Hint 2: Iterating over either columns or rows, and for each possible class label, calculate the relative
#         frequency of the observed feature value given the class (using feature_class_value_counts and 
#         feature_class_counts) 
#
# Hint 3: Calculate the non-normalized estimated class probabilities by multiplying the class priors to the
#         product of the relative frequencies
#
# Hint 4: Normalize the probabilities by dividing by the sum of the non-normalized probabilities; in case
#         this sum is zero, then set the probabilities to the class priors
#
# Hint 5: To clarify the assignment text a little: self.feature_class_value_counts should be a mapping from 
#         a column name (a specific feature) to another mapping, which given a class label and a value for 
#         the feature, returns the number of training instances which have included this combination, 
#         i.e., the number of training instances with both the specific class label and this value on the feature.
#
# Hint 6: As an additional hint, you may take a look at the slides from the NumPy and pandas lecture, to see how you 
#         may use "groupby" in combination with "size" to get the counts for combinations of values from two columns.


class NaiveBayes:

    def __init__(self):
        self.column_filter = None
        self.binning = None
        self.labels = None
        self.class_priors = None
        self.feature_class_value_counts = None
        self.feature_class_counts = None

    def fit(self, df, nobins=10, bintype="equal-width"):
        df1 = df.copy()
        df1, self.column_filter = create_column_filter(df1)
        df1, self.binning = create_bins(df1, nobins, bintype)
        self.class_priors = dict(df1['CLASS'].value_counts(normalize=True))
        self.labels = df1['CLASS'].astype('category').cat.categories.tolist()
        dict_count = {}
        dict_values_count = {}

        # populate the dictionaries (hint 5 and 6)
        for col in df1.columns:
            if col not in ['CLASS', 'ID']:
                dict_values_count[col] = df1.groupby(['CLASS', col]).size().to_dict()
                df1_tmp = df1.dropna(axis = 0,subset = ['CLASS', col])
                dict_count[col] = df1_tmp.loc[:, 'CLASS'].value_counts().to_dict()

        self.feature_class_value_counts = dict_values_count
        self.feature_class_counts = dict_count

    def predict(self, df):
        df1 = df.copy()
        df1 = apply_column_filter(df1, self.column_filter)
        df1 = apply_bins(df1, self.binning)
        df1 = df1.drop(columns = ['CLASS', 'ID'], axis=1)

        # row, columns and label for test dataset
        num_rows = df1.shape[0]
        num_columns = df1.shape[1]
        num_labels = len(self.labels)
        matrix = np.zeros([num_labels, num_rows, num_columns])

        # create a matrix with a coefficeint that is the relative frequency (hint 2)
        for col in range(num_columns):
            curr_col = df1.columns[col]
            
            for label in range(num_labels):
                curr_label = self.labels[label]

                for row in range(num_rows):
                    curr_value = df1.iloc[row, col]
                    # if the tuple (label, values) is in the dictionary, we can calculate the relative frequency
                    if (curr_label, curr_value) in self.feature_class_value_counts[curr_col].keys():
                        feature_value_count = self.feature_class_value_counts[curr_col][(curr_label, curr_value)]
                        feature_count = self.feature_class_counts[curr_col][curr_label]
                        rel_freq = feature_value_count / feature_count
                    else:
                        rel_freq = 0
                    
                    matrix[label, row, col] = rel_freq
        
        # hint 3 
        # we multiply the values of the matrix to obtain the numerator of the bayes theorem
        # the result will give us for a tuple (col, row), the relative freq given the class
        non_norm_matrix = matrix.prod(axis=2)
        # store all the classes in a np.array
        class_vector = np.array([self.class_priors[self.labels[i]] for i in range(num_labels)])
        # create a matrix of classes and then transpose it
        class_matrix = np.tile(class_vector, num_rows).reshape([num_rows, num_labels]).T
        # multiply the class matrix with relative freq matrix
        non_norm_matrix = non_norm_matrix * class_matrix

        # normalization of the matrix -> relative frequencies must sum to 1
        normalization = np.sum(non_norm_matrix, axis=0)
        # Need to handle the case where the sum is zero
        # We put the values to 1, that won't change anything but solve the problem
        # And we store the values which are zero to later substitute them
        normalizing_matrix = np.tile(normalization, num_labels).reshape([num_labels, num_rows])
        normalizing_matrix_zero = normalizing_matrix==0
        normalizing_matrix += normalizing_matrix_zero.astype('float')

        # normalize to get the final matrix (hint 4)
        result_matrix = non_norm_matrix / normalizing_matrix
        # substitute the values which were zero with the class priors
        # we add them since adding is equal to replacing if the original value was zero
        normalizing_adding_priors = normalizing_matrix_zero.astype('float')*class_matrix
        result_matrix += normalizing_adding_priors

        result_df = pd.DataFrame(result_matrix.T, columns=self.labels)
        return result_df



def check_result_bayes():
    glass_train_df = pd.read_csv("glass_train.csv")

    glass_test_df = pd.read_csv("glass_test.csv")

    nb_model = NaiveBayes()

    test_labels = glass_test_df["CLASS"]

    nobins_values = [3,5,10]
    bintype_values = ["equal-width","equal-size"]
    parameters = [(nobins,bintype) for nobins in nobins_values for bintype in bintype_values]

    results = np.empty((len(parameters),3))

    for i in range(len(parameters)):
        t0 = time.perf_counter()
        nb_model.fit(glass_train_df,nobins=parameters[i][0],bintype=parameters[i][1])
        print("Training time {0}: {1:.2f} s.".format(parameters[i],time.perf_counter()-t0))
        t0 = time.perf_counter()
        predictions = nb_model.predict(glass_test_df)
        print("Testing time {0}: {1:.2f} s.".format(parameters[i],time.perf_counter()-t0))
        results[i] = [accuracy(predictions,test_labels),brier_score(predictions,test_labels),
                    auc(predictions,test_labels)] # Assuming that you have defined auc - remove otherwise

    results = pd.DataFrame(results,index=pd.MultiIndex.from_product([nobins_values,bintype_values]),
                        columns=["Accuracy","Brier score","AUC"])

    print()
    display("results",results)

    train_labels = glass_train_df["CLASS"]
    nb_model.fit(glass_train_df)
    predictions = nb_model.predict(glass_train_df)
    print("Accuracy on training set: {0:.4f}".format(accuracy(predictions,train_labels)))
    print("AUC on training set: {0:.4f}".format(auc(predictions,train_labels)))
    print("Brier score on training set: {0:.4f}".format(brier_score(predictions,train_labels)))



if __name__ == "__main__":
    print("KNN")
    check_result_KNN()
    print("Naive Bayes")
    check_result_bayes()






        




    
