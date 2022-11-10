import pandas as pd
import numpy as np


def brier_score(df, correctlabels):
    for id in df.iterrows():
        for col in correctlabels:
            for p in id:
                obs = id[col]
                brier_score = (p-obs)**2
    
    return brier_score





def test():
    predictions = pd.DataFrame({"A":[0.5,0.5,0.5,0.25,0.25],"B":[0.5,0.25,0.25,0.5,0.25],"C":[0.0,0.25,0.25,0.25,0.5]})

    correctlabels = ["B","A","B","B","C"]

    print("Brier score: {}".format(brier_score(predictions,correctlabels)))

test()




                




