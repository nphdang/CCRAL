import numpy as np
from sklearn import metrics
from sklearn.metrics import confusion_matrix

def perf(y_test, y_pred):
    y_pred = np.around(y_pred)
    AUC = metrics.roc_auc_score(y_test, y_pred)
    F1 = metrics.f1_score(y_test, y_pred, average="macro")
    Accuracy = metrics.accuracy_score(y_test, y_pred)
    Precision = metrics.precision_score(y_test, y_pred)
    Recall = metrics.recall_score(y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    Specificity = tn / (tn+fp)
    ret = [F1, Precision, Recall, Specificity, Accuracy, AUC]
    ret = [round(e, 4) for e in ret]

    return ret

