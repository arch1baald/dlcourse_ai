import numpy as np


def binary_classification_metrics(prediction, ground_truth):
    '''
    Computes metrics for binary classification

    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels

    Returns:
    precision, recall, f1, accuracy - classification metrics
    '''
    precision = 0
    recall = 0
    accuracy = 0
    f1 = 0

    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score
    num_samples = ground_truth.shape[0]
    joint = np.hstack([ground_truth.reshape(-1, 1), prediction.reshape(-1, 1)])
    TN = np.sum((joint[:, 0] == 0) & (joint[:, 1] == 0))
    FP = np.sum((joint[:, 0] == 0) & (joint[:, 1] == 1))
    FN = np.sum((joint[:, 0] == 1) & (joint[:, 1] == 0))
    TP = np.sum((joint[:, 0] == 1) & (joint[:, 1] == 1))
    precision = TP / (FP + TP)
    recall = TP / (TP + FN)
    f1 = 2 * precision * recall / (precision + recall)
    accuracy = (TP + TN) / num_samples
    return precision, recall, f1, accuracy


def multiclass_accuracy(prediction, ground_truth):
    '''
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    '''
    # TODO: Implement computing accuracy
    from sklearn.metrics import confusion_matrix
    
    conf = confusion_matrix(ground_truth, prediction)
    accuracy = np.trace(conf) / np.sum(conf)
    return accuracy
