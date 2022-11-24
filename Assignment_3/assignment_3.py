import numpy as np
import pandas as pd
import time
import sklearn
from sklearn.tree import DecisionTreeClassifier

from functions_assignment_1 import *
from platform import python_version
from IPython.display import display

class RandomForest:

    def __init__(self):
        self.column_filter = None
        self.imputation = None
        self.one_hot = None
        self.labels = None
        self.model = None

    def fit(self, df, no_trees = 100):
        df1 = df.copy()
        display(df1)

        df1, self.column_filter = create_column_filter(df1)
        df1, self.imputation = create_imputation(df1)
        df1, self.one_hot = create_one_hot(df1)
        df1["CLASS"] = df1["CLASS"].astype("category")
        self.labels = list(df1["CLASS"].cat.categories)


        df1 = df.copy()
        y = df1["CLASS"].to_numpy()
        df1 = df1.drop(labels = "CLASS", axis = 1,)
        X = df1.to_numpy()

        cls = DecisionTreeClassifier(max_features="log2")

        cls.fit()


        ids = df.sample(df.shape[0], replace=True).index
        self.model = None


def testRanddomForest_1():
    # Test your code (leave this part unchanged, except for if auc is undefined)

    train_df = pd.read_csv("Assignment_3/tic-tac-toe_train.csv")

    test_df = pd.read_csv("Assignment_3/tic-tac-toe_test.csv")

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


testRanddomForest_1()