import numpy as np


def multiclass_accuracy(prediction, ground_truth):
    """
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    """
    from sklearn.metrics import confusion_matrix
    
    conf = confusion_matrix(ground_truth, prediction)
    accuracy = np.trace(conf) / np.sum(conf)
    return accuracy
