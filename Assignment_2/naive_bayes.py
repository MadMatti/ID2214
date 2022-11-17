import numpy as np
import pandas as pd
import time
from platform import python_version
from functions_assignment_1 import *

print(f"Python version: {python_version()}")
print(f"NumPy version: {np.__version__}")
print(f"Pandas version: {pd.__version__}")
# Define the class kNN with three functions __init__, fit and predict (after the comments):
#
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
        
        df1, self.column_filter = create_column_filter(df)
        df1, self.binning = create_bins(df, nobins, bintype)
        
      
        df1["CLASS"] = df1["CLASS"].astype("category")
        self.labels = df1["CLASS"].cat.categories
        
        classes, counts = np.unique(df1["CLASS"], return_counts=True)
        self.class_priors = {classes[i]: counts[i]/len(classes) for i in range(len(classes))}
        
        df1 = df1.drop(axis = 1, labels = ["CLASS", "ID"])
        features = df1.columns
        
       
              
                    
                
        
        





# Test your code (leave this part unchanged, except for if auc is undefined)
def test():
    
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
    
    return