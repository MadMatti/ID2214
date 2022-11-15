import numpy as np
import pandas as pd
import time
from functions_assignment_1 import *
from platform import python_version

print(f"Python version: {python_version()}")
print(f"NumPy version: {np.__version__}")
print(f"Pandas version: {pd.__version__}")
class NaiveBayes:
    def __init__(self):
        column_filter = None
        binning = None
        labels = None
        class_priors = None
        feature_class_value_counts = None
        feature_class_counts = None
    

    def fit(self, df, nobins=10, bintype="equal-width"):
        df1 = df.copy()
        print(df1)
        
        df, self.column_filter = create_column_filter(df1)
        df, self.binning = create_bins(df1, nobins, bintype)
        self.class_priors = {c:sum(df["CLASS"]==c) for c in pd.unique(df["CLASS"])}
        df1["CLASS"] = df1["CLASS"].astype("category")
        self.labels = list(df1["CLASS"].cat.categories)
        
        for column in df.columns:
            if column not in ["CLASS", "ID"]:
                temp = df1.dropna(axis = 0, how="any", subset = ["CLASS", column])
                self.feature_class_counts = {column:len(temp["CLASS"])}
        self.feature_class_value_counts = {c:sum(df["CLASS"]==c) for c in pd.unique(df["CLASS"])}
        return None

        # For numerical stability calculate the log of a priori probability + log of each conditional probability.
        # 
    def predict(self, df):
        df1 = df.copy()
        df1 = apply_column_filter(df1)
        df1 = apply_bins(df1)
        return None

def test():
    # Test your code (leave this part unchanged, except for if auc is undefined)

    glass_train_df = pd.read_csv("Assignment_2/glass_train.csv")

    glass_test_df = pd.read_csv("Assignment_2/glass_test.csv")

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

test()