import numpy as np
import pandas as pd
from sklearn import metrics

def false_positive_rate(y_true, y_pred):
    """
    Calculate the false positive rate (FPR) for a binary classification model.

    Parameters:
    - y_true (array-like): The true labels of the binary classification problem.
    - y_pred (array-like): The predicted labels of the binary classification problem.

    Returns:
    - float: The false positive rate (FPR) calculated as the ratio of false positives to the sum of false positives and true negatives.
             If the sum of false positives and true negatives is zero, the FPR is set to zero.
    """
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()
    return fp / (fp + tn) if (fp + tn) != 0 else 0


def false_negative_rate(y_true, y_pred):
    """
    Calculate the false negative rate (FNR) for a binary classification problem.

    Parameters:
    - y_true (array-like): The true labels of the binary classification problem.
    - y_pred (array-like): The predicted labels of the binary classification problem.

    Returns:
    - float: The false negative rate (FNR) calculated as the ratio of false negatives to the sum of false negatives and true positives.
             If the sum of false negatives and true positives is zero, the FNR is set to zero.
    """
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()
    return fn / (fn + tp) if (fn + tp) != 0 else 0


def auc_precision_recall(y_true, y_pred):
    """
    Calculate the area under the precision-recall curve (AUC-PR).

    Parameters:
    - y_true (array-like): True binary labels.
    - y_pred (array-like): Target scores, can either be probability estimates of the positive class or confidence values.

    Returns:
    - float: The area under the precision-recall curve.

    Note:
    - The precision-recall curve is a plot of the precision (y-axis) and the recall (x-axis) for different thresholds.
    - AUC-PR summarizes the integral or the area under the precision-recall curve.
    - AUC-PR ranges from 0 to 1, with 1 indicating perfect precision and recall, and 0 indicating the worst performance.
    """
    precision, recall, _ = metrics.precision_recall_curve(y_true, y_pred)
    return metrics.auc(recall, precision)


def compute_metrics(y_true, y_score, threshold=0.5):
    """
    Compute various evaluation metrics for a binary classification model.

    Parameters:
    - y_true (array-like): The true labels of the binary classification problem.
    - y_score (array-like): The predicted scores or probabilities of the positive class.
    - threshold (float, optional): The threshold value to determine the predicted labels. Default is 0.5.

    Returns:
    - dict: A dictionary containing the following evaluation metrics:
        - accuracy (float): The accuracy of the model.
        - balanced_accuracy (float): The balanced accuracy of the model.
        - recall (float): The recall (sensitivity) of the model.
        - precision (float): The precision of the model.
        - f1 (float): The F1 score of the model.
        - fpr (float): The false positive rate (FPR) of the model.
        - fnr (float): The false negative rate (FNR) of the model.
        - pr_auc (float): The area under the precision-recall curve (AUC-PR) of the model.
        - roc_auc (float): The area under the receiver operating characteristic curve (AUC-ROC) of the model.
        - avg_acc_recall (float): The average of accuracy and recall.

    Note:
    - The accuracy is the ratio of correctly predicted samples to the total number of samples.
    - The balanced accuracy is the average of recall obtained on each class.
    - The recall is the ratio of true positives to the sum of true positives and false negatives.
    - The precision is the ratio of true positives to the sum of true positives and false positives.
    - The F1 score is the harmonic mean of precision and recall.
    - The false positive rate (FPR) is the ratio of false positives to the sum of false positives and true negatives.
    - The false negative rate (FNR) is the ratio of false negatives to the sum of false negatives and true positives.
    - The area under the precision-recall curve (AUC-PR) summarizes the integral or the area under the precision-recall curve.
    - The area under the receiver operating characteristic curve (AUC-ROC) summarizes the integral or the area under the receiver operating characteristic curve.
    - The average of accuracy and recall (avg_acc_recall) provides a combined measure of model performance.
    """
    y_pred = y_score >= threshold
    result_metrics = dict(
        accuracy = metrics.accuracy_score(y_true, y_pred),
        balanced_accuracy = metrics.balanced_accuracy_score(y_true, y_pred),
        recall = metrics.recall_score(y_true, y_pred),
        precision = metrics.precision_score(y_true, y_pred),
        f1 = metrics.f1_score(y_true, y_pred),
        fpr = false_positive_rate(y_true, y_pred),
        fnr = false_negative_rate(y_true, y_pred),
        pr_auc = auc_precision_recall(y_true, y_pred),
        roc_auc = metrics.roc_auc_score(y_true, y_score)
    )
    result_metrics['avg_acc_recall'] = np.mean([result_metrics['accuracy'], result_metrics['recall']])
    return result_metrics



def prepare_cv_results(cv_results: dict) -> pd.DataFrame:
    """
    Prepare cross-validation results from a machine learning model.

    Parameters:
    - cv_results (dict): A dictionary containing the cross-validation results.

    Returns:
    - pd.DataFrame: A pandas DataFrame containing the cross-validation results, including train metrics (if available), test metrics, and time metrics.
    """
    train_metrics_available = any(['train_' in k for k in cv_results])

    df_cv_time = pd.DataFrame({k.split('_time')[0]:v for k,v in cv_results.items() if 'time' in k})
    df_cv_time.columns = pd.MultiIndex.from_tuples([('Time', col) for col in df_cv_time.columns])

    df_cv_test = pd.DataFrame({k.split('test_')[1]:v for k,v in cv_results.items() if k.startswith('test_')})
    df_cv_test.columns = pd.MultiIndex.from_tuples([('Test', col) for col in df_cv_test.columns])

    if train_metrics_available:

        df_cv_train = pd.DataFrame({k.split('train_')[1]:v for k,v in cv_results.items() if k.startswith('train_')})
        df_cv_train.columns = pd.MultiIndex.from_tuples([('Train', col) for col in df_cv_train.columns])
        df_cv = pd.concat([df_cv_train,df_cv_test, df_cv_time], axis=1).rename_axis('k')
    else:
        df_cv = pd.concat([df_cv_test, df_cv_time], axis=1).rename_axis('k')

    df_cv = pd.concat([df_cv_train,df_cv_test, df_cv_time], axis=1).rename_axis('k')
    df_cv.index+=1

    return df_cv
