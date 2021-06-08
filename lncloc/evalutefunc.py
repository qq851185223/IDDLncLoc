import numpy as np


def metric(cnf_matrix):
    FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    TP = np.diag(cnf_matrix)
    TN = cnf_matrix.sum() - (FP + FN + TP)

    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)
    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP / (TP + FN)
    # Specificity or true negative rate
    TNR = TN / (TN + FP)
    # Precision or positive predictive value
    # PPV = TP / (TP + FP)
    # Negative predictive value
    # NPV = TN / (TN + FN)
    # Fall out or false positive rate
    # FPR = FP / (FP + TN)
    # False negative rate
    # FNR = FN / (TP + FN)
    # False discovery rate
    # FDR = FP / (TP + FP)
    tmp = (TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)
    MCC = (TP * TN - FP * FN) / (tmp ** 0.5)
    # Overall accuracy
    ACC = (TP + TN) / (TP + FP + FN + TN)
    return TPR, TNR, MCC
