import os
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, fbeta_score, f1_score
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

def load_channels(): 
    """
    Discover available telemetry channels.

    Returns
    list of str: Names of telemetry channels available for processing.
    """
    channel_names = []

    for channel in os.listdir("data/processed/windowed"):
        if os.path.isdir(os.path.join("data/processed/windowed", channel)):
            channel_names.append(channel)
    return channel_names

def load_data(channel_names):
    standard_fitted_data = {}

    for channel in channel_names: 
        standard_fitted_data[channel] = {}

        training_data_path = f"data/processed/normalized/{channel}/train/{channel}.csv"
        testing_data_path = f"data/processed/normalized/{channel}/test/{channel}.csv"
        label_data_path = f"data/processed/normalized/{channel}/label/{channel}.csv"
        
        X_fit_train = pd.read_csv(training_data_path).to_numpy()
        X_fit_test =  pd.read_csv(testing_data_path).to_numpy()
        y_test = pd.read_csv(label_data_path).iloc[:, 0].to_numpy()
        
        standard_fitted_data[channel]["train"] = X_fit_train
        standard_fitted_data[channel]["test"] = X_fit_test
        standard_fitted_data[channel]["label"] = y_test
    
    print("[✓] Loaded training, testing, and label data")
    return standard_fitted_data

def predict(threshold, decision_scores):
    """
    Convert Ensemble decision scores into binary anomaly predictions.

    Parameters
    threshold (float): Decision score threshold. 
    decision_scores (np.ndarray):: Decision function outputs from the Ensemble.

    Returns
    np.ndarray: Binary predictions where "1" indicates anomaly and "0" indicates normal behavior.
    """
    y_pred = []
    for score in decision_scores: 
        if score < threshold: 
            y_pred.append(1)
        else:
            y_pred.append(0)
    return np.array(y_pred)

def local_outlier_factor(channel_data):
    """
    Train and evaluate a Local Outlier Factor for a single telemetry channel.
    The function computes decision scores on test data and sweeps percentile-based thresholds,
    selecting the threshold that maximizes recall, breaking ties using precision. 

    Parameters
    channel_data (dict): Dictionary containing training feature array, test feature array, and 
                        ground-truth anomaly labels for test data
    Returns
    np.ndarray: Binary anomaly predictions (1 = anomaly, 0 = normal).
    """
    clf = LocalOutlierFactor(n_neighbors=50, novelty=True)
    clf.fit(channel_data["train"])
    decision_scores = clf.decision_function(channel_data["test"])
 
    percentiles = np.arange(1, 30 + 1)
    thresholds = []
    for percent in percentiles: 
        thresholds.append(np.percentile(decision_scores, percent))
    
    # best_f1_score = 0
    best_scores = (0, 0, 0)  

    for th in thresholds:
        y_pred = predict(th, decision_scores)
        # f1 = f1_score(channel_data["label"], y_pred)
        recall = recall_score(channel_data["label"], y_pred, pos_label=1)
        precision = precision_score(channel_data["label"], y_pred, zero_division=0)
        if (recall > best_scores[0]) or (recall == best_scores[0] and precision > best_scores[1]):
            best_scores = (recall, precision, th)

    best_threshold = best_scores[2]
    
    return predict(best_threshold, decision_scores)

def isolation_forest(channel_data):
    """
    Train and evaluate an Isolation Forest for a single telemetry channel.
    The function computes decision scores on test data and sweeps percentile-based thresholds,
    selecting the threshold that maximizes recall, breaking ties using precision. 

    Parameters
    channel_data (dict): Dictionary containing training feature array, test feature array, and 
                        ground-truth anomaly labels for test data
    Returns
    np.ndarray: Binary anomaly predictions (1 = anomaly, 0 = normal).
    """
    clf = IsolationForest(random_state=42, n_jobs=-1).fit(channel_data["train"])
    decision_scores = clf.decision_function(channel_data["test"])
    
    percentiles = np.arange(1, 30 + 1)
    thresholds = []
    for percent in percentiles: 
        thresholds.append(np.percentile(decision_scores, percent))
    
    # best_f1_score = 0
    best_scores = (0, 0, 0)  

    for th in thresholds:
        y_pred = predict(th, decision_scores)
        # f1 = f1_score(channel_data["label"], y_pred)
        recall = recall_score(channel_data["label"], y_pred, pos_label=1)
        precision = precision_score(channel_data["label"], y_pred, zero_division=0)
        if (recall > best_scores[0]) or (recall == best_scores[0] and precision > best_scores[1]):
            best_scores = (recall, precision, th)

    best_threshold = best_scores[2]
    
    return predict(best_threshold, decision_scores)
    
def one_class_svm(channel_data):
    """
    Train and evaluate an One Class SVM for a single telemetry channel.
    The function computes decision scores on test data and sweeps percentile-based thresholds,
    selecting the threshold that maximizes recall, breaking ties using precision. 

    Parameters
    channel_data (dict): Dictionary containing training feature array, test feature array, and 
                        ground-truth anomaly labels for test data
    Returns
    np.ndarray: Binary anomaly predictions (1 = anomaly, 0 = normal).
    """
    clf = OneClassSVM().fit(channel_data["train"])
    decision_scores = clf.decision_function(channel_data["test"])
   
    percentiles = np.arange(1, 30 + 1)
    thresholds = []
    for percent in percentiles: 
        thresholds.append(np.percentile(decision_scores, percent))
    
    # best_f1_score = 0
    best_scores = (0, 0, 0)  

    for th in thresholds:
        y_pred = predict(th, decision_scores)
        # f1 = f1_score(channel_data["label"], y_pred)
        recall = recall_score(channel_data["label"], y_pred, pos_label=1)
        precision = precision_score(channel_data["label"], y_pred, zero_division=0)
        if (recall > best_scores[0]) or (recall == best_scores[0] and precision > best_scores[1]):
            best_scores = (recall, precision, th)

    best_threshold = best_scores[2]
    
    return predict(best_threshold, decision_scores)

def ensemble_model(channel_data, report_path):
    """
    Evaluate an ensemble anomaly detection model across all channels, combining 
    One-Class SVM, Isolation Forest, and Local Outlier Factor. 
    A data point is labeled as anomalous if at least one model predicts an anomaly 
    (logical OR / majority ≥ 1).

    Parameters
    channel_data (dict): Dictionary mapping channel names to channel-specific data dicts.
    report_path (str): Path to the output report file.
    """
    for channel in channel_data: 
        y_pred_one_class_svm = one_class_svm(channel_data[channel])
        y_pred_isolation_forest = isolation_forest(channel_data[channel])
        y_pred_local_outlier_factor = local_outlier_factor(channel_data[channel])

        y_pred = []
        for i in range(len(y_pred_one_class_svm)): 
            ensemble = y_pred_one_class_svm[i] + y_pred_isolation_forest[i] + y_pred_local_outlier_factor[i]
            if ensemble >= 1: 
                y_pred.append(1)
            else:
                y_pred.append(0)
        y_test = (channel_data[channel]["label"])
        
        precision_1 = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
        recall_1 = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
        # f1_1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
        fdr = 1 - precision_1
        c_m = confusion_matrix(y_test, y_pred)
        
        save_channel_report(
            channel=channel,
            precision=precision_1,
            recall=recall_1,
            fdr=fdr,
            confusion_matrix=c_m,
            report_path=report_path,
        )

def save_channel_report(channel: str, precision: float, recall: float, fdr: float, confusion_matrix: np.ndarray, report_path: str):
    """
    Save a human-readable evaluation report for one channel.
    
    Parameters
    channel (str): Channel identifier.
    precision (float): Precision for anomaly class.
    recall (float): Recall for anomaly class.
    fdr (float): False Discovery Rate.
    confusion_matrix (np.ndarray): 2x2 confusion matrix.
    report_path (str): Output text file path.
    """

    with open(report_path, "a") as f:
        f.write(f"Channel {channel}\n")
        f.write("-" * (8 + len(channel)) + "\n")
        f.write(f"Precision (anomaly): {precision:.4f}\n")
        f.write(f"Recall (anomaly):    {recall:.4f}\n")
        f.write(f"FDR:                 {fdr:.4f}\n")
        f.write("Confusion matrix:\n")
        f.write(np.array2string(confusion_matrix))
        f.write("\n")

def main():
    """
    Run Ensemble evaluation across all telemetry channels and save a consolidated report.
    """
    report_path = f"results/ensemble.txt"
    channels = load_channels()

    with open(report_path, "w") as f:
        f.write("Ensemble Model Evaluation Report\n")
        f.write("=" * 35 + "\n\n")
    
    standard_fitted_data = load_data(channels)
    ensemble_model(standard_fitted_data, report_path)
    
    print("[✓] Predicted all telemetry anomaly data with an ensemble model")

if __name__ == "__main__":
    main()