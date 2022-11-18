# One feature(label/attribute) on each axis, one for y, and one for x

# ALGORITHM
# Input: A row/instance
#       Set of training data
#       Constant K
# Euclidean distance metric
#       1. Categorical features to numerical (one_hot encoding)
#       2. replace missing values with imputation or exclude
#       3. Normalize numerical features (Does it include categorical?)
# 
# For speed Up (Maybe not applicable)
# Reduce Dimensionality
# Sampling training data or protoype selection
# Partitioning the feature space        
import numpy as np
import pandas as pd
import time
from functions_assignment_1 import *
from IPython.display import display

class kNN():
    def __init__(self):
        self.column_filter = None 
        self.imputation = None
        self.normalization = None
        self.one_hot = None
        self.labels  = None
        self.training_labels = None 
        self.training_data = None
        self.training_time = None

    def fit(self, df, normalizationtype="minmax"):
        df1 = df.copy()

        df, self.column_filter = create_column_filter(df1)
        df1, self.imputation = create_imputation(df1)
        df1, self.normalization = create_normalization(df1, normalizationtype)
        df1, self.one_hot = create_one_hot(df1)
        
        df1["CLASS"] = df1["CLASS"].astype("category")
        self.training_labels = df1["CLASS"]
        self.labels = list(self.training_labels.cat.categories)
        df1 = df1.drop(axis = 1, labels = ["CLASS", "ID"])
        self.training_data = np.array(df1)
        
        print(df1)

    def get_distance(self, x_1, x_2):
        final_distance  = np.sqrt(np.sum((x_1 - x_2)** 2)) 
        return final_distance

    def get_nearest_neighbor_predictions(self, x_test,k):
        distances = np.empty(self.training_data.shape[0])
        for index, point in enumerate(self.training_data):
            distances[index] = self.get_distance(x_test, point)

        sorted_indices = distances.argsort()
        
        neighbours = sorted_indices[:k]
        k_labels = self.training_labels[neighbours]

        unique, counts =  np.unique(k_labels, return_counts = True) 
        probabilities = dict(zip(unique, counts/k))
        
        return probabilities


    def predict(self, df, k=5):
        df1 = df.copy()
        df1 = apply_column_filter(df1, self.column_filter)
        df1 = apply_imputation(df1, self.imputation)
        df1 = apply_normalization(df1, self.normalization)
        df1 = apply_one_hot(df1, self.one_hot)

        to_train = np.unique(self.training_labels)
        df1 = df1.drop(["CLASS","ID"], axis=1)

        values = df1.select_dtypes(include=np.number).to_numpy()
        
        prob_df = pd.DataFrame(data = 0.0, index=range(len(values)), columns=to_train)
        
        for i in range(len(values)):
            row = values[i]
            probabilities = self.get_nearest_neighbor_predictions(row, k)
            
            for prob in probabilities.keys():
                prob_df.at[i, prob] = probabilities.get(prob) 
        return prob_df


        


def test():
    # Test your code (leave this part unchanged, except for if auc is undefined)

    glass_train_df = pd.read_csv("Assignment_2/glass_train.csv")

    glass_test_df = pd.read_csv("Assignment_2/glass_test.csv")

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
test()