import pandas as pd
import numpy as np


def brier_score(df, correctlabels):
    
    brier_score = 0
    n = len(df.index)
          
    for i,p in df.iterrows():
        
        true_label = correctlabels[i]

        o = np.zeros(len(p))
        o[df.columns==true_label] = 1
        
        brier_score += ((p-o)**2).sum(axis=0)/n
        
    return brier_score
        
        
        



def test():
    predictions = pd.DataFrame({"A":[0.5,0.5,0.5,0.25,0.25],"B":[0.5,0.25,0.25,0.5,0.25],"C":[0.0,0.25,0.25,0.25,0.5]})

    correctlabels = ["B","A","B","B","C"]
    

    print("Brier score: {}".format(brier_score(predictions,correctlabels)))

test()




                




