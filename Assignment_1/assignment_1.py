import pandas as pd
import numpy as np


def get_pos_and_neg(predictions, pos_label, correctlabels):
    scores = {}
    for i in range(len(predictions)):
        score = predictions[i, pos_label]
        p, n = scores.get(score,(0,0))
        if correctlabels[i] == pos_label:
            scores[score] = (p+1,n)
        else:
            scores[score] = (p,n+1)
    scores = np.array([[score,scores[score][0],scores[score][1]] for score in scores.keys()])
    scores = scores[np.argsort(scores[:,0])[::-1]]
    pos = scores[:,1]
    neg = scores[:,2]
    return scores, pos, neg




def auc(predictions, correctlabels):

    pos_label = 0

    pos, neg = get_pos_and_neg(predictions,pos_label,correctlabels)

    tpr = [cs/sum(pos) for cs in np.cumsum(pos)]
    fpr = [cs/sum(neg) for cs in np.cumsum(neg)]

    return "auc"




def test():
    predictions = pd.DataFrame({"A":[0.9,0.9,0.6,0.55],"B":[0.1,0.1,0.4,0.45]})

    correctlabels = ["A","B","B","A"]

    print("AUC: {}".format(auc(predictions,correctlabels)))
test()




                




