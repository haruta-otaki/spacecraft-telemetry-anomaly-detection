"""
MiniRocket Feature Extraction and Normalization Pipeline

This script performs feature extraction and normalization for multivariate
telemetry time-series data using MiniRocket. For each telemetry channel, the
pipeline:

1. Loads pre-windowed training and test data.
2. Fits a MiniRocketMultivariate transformer on training data only.
3. Transforms both training and test sets into feature space.
4. Persists the fitted MiniRocket model to disk for reproducibility.
5. Applies z-score normalization using a StandardScaler fitted on training data.
6. Persists the fitted scaler and stores normalized features as CSV files.

All learned artifacts (MiniRocket transformers and scalers) are saved to the
`artifacts/` directory, while transformed datasets are written to `data/processed/`.
"""

import os
import argparse
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sktime.transformations.panel.rocket import MiniRocketMultivariate

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

def transform_data(channel_names, kernels, jobs):
    """
    Apply MiniRocket feature extraction to windowed telemetry data,
    transforming both training and test data into feature space and writing the 
    CSV files to data/processed/transformed/{channel}/ as well as the MiniRocket models
    to artifacts/minirocket/{channel}.pkl.

    Parameters
    channel_names (list of str): List of telemetry channel identifiers.
    kernels (int): Number of convolutional kernels used by MiniRocket.
    jobs (int): Number of CPU cores used for parallel feature extraction.
    """
    transformed_data= {}
    os.makedirs("data/processed/transformed", exist_ok=True)
    os.makedirs("artifacts/minirocket", exist_ok=True)

    for channel in channel_names:
        transformed_data[channel] = {}
        training_data_path = f"data/processed/windowed/{channel}/train/{channel}.npy"
        testing_data_path = f"data/processed/windowed/{channel}/test/{channel}.npy"
        label_data_path = f"data/processed/windowed/{channel}/label/{channel}.npy"
        
        X_train = np.load(training_data_path)
        X_test = np.load(testing_data_path)
        y_test = np.load(label_data_path)

        minirocket = MiniRocketMultivariate(num_kernels=kernels, n_jobs = jobs, random_state = 42) 
        minirocket.fit(X_train)
        X_transform_train = minirocket.transform(X_train)
        X_transform_test = minirocket.transform(X_test)

        transformed_data[channel]["train"] = X_transform_train
        transformed_data[channel]["test"] = X_transform_test
        transformed_data[channel]["label"] = y_test

    for channel in transformed_data: 
        X_transform_train = transformed_data[channel]["train"]
        X_transform_test = transformed_data[channel]["test"]
        y_test = transformed_data[channel]["label"]

        train_df = pd.DataFrame(X_transform_train)
        test_df = pd.DataFrame(X_transform_test)
        label_df = pd.DataFrame(y_test)
        training_data_path = f"data/processed/transformed/{channel}/train"
        testing_data_path = f"data/processed/transformed/{channel}/test"
        label_data_path = f"data/processed/transformed/{channel}/label"
        
        joblib.dump(minirocket, f"artifacts/minirocket/{channel}.pkl")
        os.makedirs(training_data_path, exist_ok=True)
        os.makedirs(testing_data_path, exist_ok=True)
        os.makedirs(label_data_path, exist_ok=True)

        train_df.to_csv(training_data_path + f"/{channel}.csv", index=False)
        test_df.to_csv(testing_data_path + f"/{channel}.csv", index=False)
        label_df.to_csv(label_data_path + f"/{channel}.csv", index=False)  
    print("[✓] Applied MiniRocket feature extraction")     

def normalize_data(channel_names):
    """
    Normalize MiniRocket features using z-score normalization.
    The normalized feature CSVs are written to data/processed/normalized/{channel}/ and 
    the fitted scalers are written to artifacts/scaler/{channel}.pkl.

    Parameters
    channel_names (list of str): List of telemetry channel identifiers.
    """
    standard_fitted_data = {}
    os.makedirs("data/processed/normalized", exist_ok=True)
    os.makedirs("artifacts/scaler", exist_ok=True)

    for channel in channel_names: 
        standard_fitted_data[channel] = {}

        training_data_path = f"data/processed/transformed/{channel}/train/{channel}.csv"
        testing_data_path = f"data/processed/transformed/{channel}/test/{channel}.csv"
        label_data_path = f"data/processed/transformed/{channel}/label/{channel}.csv"
        
        X_transform_train = pd.read_csv(training_data_path).to_numpy()
        X_transform_test =  pd.read_csv(testing_data_path).to_numpy()
        y_test = pd.read_csv(label_data_path).iloc[:, 0].to_numpy()

        scaler = StandardScaler()
        scaler.fit(X_transform_train)
        X_fit_train = scaler.transform(X_transform_train)
        X_fit_test = scaler.transform(X_transform_test)
        
        standard_fitted_data[channel]["train"] = X_fit_train
        standard_fitted_data[channel]["test"] = X_fit_test
        standard_fitted_data[channel]["label"] = y_test
        

    for channel in standard_fitted_data: 
        X_fit_train = standard_fitted_data[channel]["train"]
        X_fit_test = standard_fitted_data[channel]["test"]
        y_test = standard_fitted_data[channel]["label"]

        train_df = pd.DataFrame(X_fit_train)
        test_df = pd.DataFrame(X_fit_test)
        label_df = pd.DataFrame(y_test)

        training_data_path = f"data/processed/normalized/{channel}/train"
        testing_data_path = f"data/processed/normalized/{channel}/test"
        label_data_path = f"data/processed/normalized/{channel}/label"
        
        joblib.dump(scaler, f"artifacts/scaler/{channel}.pkl")
        os.makedirs(training_data_path, exist_ok=True)
        os.makedirs(testing_data_path, exist_ok=True)
        os.makedirs(label_data_path, exist_ok=True)

        train_df.to_csv(training_data_path + f"/{channel}.csv", index=False)
        test_df.to_csv(testing_data_path + f"/{channel}.csv", index=False)
        label_df.to_csv(label_data_path + f"/{channel}.csv", index=False)  
    print("[✓] Normalized data")     


def parse_args():
    """
    Parse command-line arguments for MiniRocket configuration.

    Returns
    argparse.Namespace: parsed arguments containing number of MiniRocket kernels
                        and the number of CPU cores for parallel execution.
    """
    parser = argparse.ArgumentParser(description="Initialize hyper parameters for MiniRocket")
    parser.add_argument(
        "--kernels",
        type=int,
        default=10000,
        help="number of kernels for MiniRocket (default: 10000)"
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=2,
        help="number of cores used for MiniRocket (default: 2)"
    )
    return parser.parse_args()

def main():
    """
    Execute the MiniRocket preprocessing pipeline.

    This function:
    1. Parses command-line arguments.
    2. Discovers available telemetry channels.
    3. Applies MiniRocket feature extraction.
    4. Applies feature normalization.
    """
    args = parse_args()
    channels = load_channels()
    kernels = args.kernels
    jobs = args.jobs
    transform_data(channels, kernels, jobs)
    normalize_data(channels)

if __name__ == "__main__":
    main()