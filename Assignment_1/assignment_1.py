import pandas as pd
import numpy as np


# def get_pos_and_neg(predictions, pos_label, correctlabels):
#     scores = {}
#     for i in range(len(predictions)):
#         score = predictions[i, pos_label]
#         p, n = scores.get(score,(0,0))
#         if correctlabels[i] == pos_label:
#             scores[score] = (p+1,n)
#         else:
#             scores[score] = (p,n+1)
#     scores = np.array([[score,scores[score][0],scores[score][1]] for score in scores.keys()])
#     scores = scores[np.argsort(scores[:,0])[::-1]]
#     pos = scores[:,1]
#     neg = scores[:,2]
#     return scores, pos, neg




# def auc(predictions, correctlabels):

#     pos_label = 0

#     pos, neg = get_pos_and_neg(predictions,pos_label,correctlabels)

#     tpr = [cs/sum(pos) for cs in np.cumsum(pos)]
#     fpr = [cs/sum(neg) for cs in np.cumsum(neg)]

    # return "auc"


def auc(df, correctlabels):
    thresholds = list(np.array(list(range(0, 105, 1)))/100)
    roc_point = []

    for threshold in thresholds:
        tp =0; fp=0; fn=0; tn=0

        for index, instance in df.iterrows():
            actual = instance["actual"]
            prediction = instance["prediction"]

            if(prediction >= threshold):
                prediction_class = 1
            else:
                prediction_class = 0
            if prediction_class == 1 and actual ==1:
                tp += 1
            elif actual == 1 and prediction_class == 0:
                fn += 1
            elif actual == 0 and prediction_class == 1:
                fp += 1
            elif actual == 0 and prediction_class == 0:
                tn += 1

            tpr = tp / (tp+fn)
            fpr = fp / (tn+fp)

            print(tp, fp, fn, tn)
            roc_point.append([tpr, fpr])
    pivot = pd.DataFrame(roc_point, columns = ["x","y"])
    pivot["thresholds"] = thresholds
    auc_score = auc_score(pivot)
    return auc_score

def auc_score(pivot):
    return abs(np.trap(pivot.x, pivot.y))

def test():
    predictions = pd.DataFrame({"A":[0.9,0.9,0.6,0.55],"B":[0.1,0.1,0.4,0.45]})

    correctlabels = ["A","B","B","A"]

    print("AUC: {}".format(auc(predictions,correctlabels)))
test()




                




