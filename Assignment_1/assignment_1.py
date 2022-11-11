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
    auc_score = 0
    threshold_values = []
    roc_point = []
    dummie = pd.get_dummies(correctlabels)

    for i in range(0, len(df.axes[1])):
        if i == 0: 
            df1 = pd.DataFrame({"actual": dummie.iloc[:, i] , "prediction":df.iloc[:, i]})
        if i != 0:
            df2 = pd.DataFrame({"actual": dummie.iloc[:, i] , "prediction":df.iloc[:, i]})
            df1 = pd.concat([df1, df2], ignore_index = True)
    print('This is the datafram: \n',df1)
        
    for threshold in thresholds:
        tp=0; fp=0; fn=0; tn=0

        for index, instance in df1.iterrows():
            # actual = np.sum((instance*dummie.iloc[index]))
            actual = instance["actual"]
            prediction = instance["prediction"]

            if(prediction >= threshold):
                prediction_class = 1
            else:
                prediction_class = 0
            if actual == 1 and prediction_class == 1:
                tp += 1
            elif actual == 1 and prediction_class == 0:
                fn += 1
            elif actual == 0 and prediction_class == 1:
                fp += 1
            elif actual == 0 and prediction_class == 0:
                tn += 1

        # print(tp, fp, fn, tn)
        tpr = tp / (tp+fn)
        fpr = fp / (tn+fp)
        # print(tpr, fpr)
        roc_point.append([tpr, fpr])
        # print(roc_point)
        threshold_values.append(threshold) 
    pivot = pd.DataFrame(roc_point, columns = ["x","y"])
    pivot["threshold"] = threshold_values
    print(pivot)
    auc_score = abs(np.trapz(pivot.x, pivot.y))
    return auc_score

# def auc_score(pivot):
#     return int()

def test():
    predictions = pd.DataFrame({"A":[0.5,0.5,0.5,0.25,0.25],"B":[0.5,0.25,0.25,0.5,0.25],"C":[0.0,0.25,0.25,0.25,0.5]})

    correctlabels = ["B","A","B","B","C"]

    print("AUC: {}".format(auc(predictions,correctlabels)))
test()




                




