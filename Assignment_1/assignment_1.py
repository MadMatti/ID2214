import pandas as pd
import numpy as np


def brier_score(df, correctlabels):

    obs_vector = []

    for label in correctlabels:
        index = df[label]
        print(index)






def test():
    predictions = pd.DataFrame({"A":[0.5,0.5,0.5,0.25,0.25],"B":[0.5,0.25,0.25,0.5,0.25],"C":[0.0,0.25,0.25,0.25,0.5]})

    correctlabels = ["B","A","B","B","C"]

    print("Brier score: {}".format(brier_score(predictions,correctlabels)))

test()




                




