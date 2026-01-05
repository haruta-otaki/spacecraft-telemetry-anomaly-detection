"""
Telemetry Preprocessing and Sliding Window Generation

This module implements the preprocessing pipeline for telemetry anomaly detection.
It converts raw NumPy telemetry streams into windowed datasets suitable for
unsupervised and semi-supervised anomaly detection models.

Pipeline overview:
1. Load raw telemetry data stored in .npy format
2. Filter channels by spacecraft (SMAP)
3. Convert raw arrays to CSV for inspection and reproducibility
4. Group telemetry streams by channel
5. Apply sliding window segmentation with configurable size and overlap
6. Assign window-level anomaly labels on test data based on annotated anomaly intervals
7. Persist processed datasets as NumPy arrays

Example usage:
    python src/preprocessing/feature_selection.py --size 10000 --overlap 2
"""

from pathlib import Path
import os
import ast
import argparse
import pandas as pd
import numpy as np

def load_data(): 
    """
    Load raw telemetry data from NumPy files and convert them to CSV format
    whilst filtering channels belonging to the SMAP spacecraft. 
    
    Output CSV files to:
        data/raw/csv_format/train/
        data/raw/csv_format/test/
    """
    
    training_dfs = []
    test_dfs = []

    training_df = []
    test_df = []

    file_names = []

    label_df = pd.read_csv("data/raw/npy_format/labeled_anomalies.csv")

    train_data_path = "data/raw/npy_format/train" 
    test_data_path = "data/raw/npy_format/test" 
    
    os.makedirs("data/raw/csv_format/train", exist_ok=True)
    os.makedirs("data/raw/csv_format/test", exist_ok=True)

    index = 0
    for root, _, files in os.walk(train_data_path):
        for file in files:
            example_path = os.path.join(root, file)
            filename = Path(example_path).stem
            
            row_indices = label_df[label_df['chan_id'] == filename].index.tolist()
            if not row_indices:
                continue
            spacecraft = label_df.loc[row_indices[0], 'spacecraft']
            
            if (spacecraft == "SMAP"): 
                data = np.load(example_path)
                df = pd.DataFrame(data)
                df.to_csv(f"data/raw/csv_format/train/{filename}.csv", index=False)
                training_dfs.append(df) 
                index += 1

    index = 0
    for root, _, files in os.walk(test_data_path):
        for file in files:
            example_path = os.path.join(root, file)
            filename = Path(example_path).stem
            
            row_indices = label_df[label_df['chan_id'] == filename].index.tolist()
            if not row_indices:
                continue
            spacecraft = label_df.loc[row_indices[0], 'spacecraft']
            
            if (spacecraft == "SMAP"): 
                data = np.load(example_path)
                df = pd.DataFrame(data)
                df.to_csv(f"data/raw/csv_format/test/{filename}.csv", index=False)
                test_dfs.append(df) 
                file_names.append(filename)
                
                index += 1
    print("[✓] Converted raw .npy files to .csv files")

def retreive_data():
    """
    Load preprocessed CSV telemetry data and organize it by channel where 
    channel types are inferred from the first character of the file name.

    Returns:
        training_dfs (dict): Mapping from channel ID to a list of training DataFrames.
        test_dfs (dict): Mapping from channel ID to a list of test DataFrames.
        label_df (pd.DataFrame): DataFrame containing anomaly annotations.
        file_names (dict): Mapping of channel IDs to their corresponding file identifiers.
    """
        
    training_dfs = {}
    test_dfs = {}
    file_names = {"train": {}, "test": {}}
    label_df = pd.read_csv("data/raw/npy_format/labeled_anomalies.csv")
    train_data_path = "data/raw/csv_format/train" 
    test_data_path = "data/raw/csv_format/test" 
    
    for root, _, files in os.walk(train_data_path):
        for file in files:
            example_path = os.path.join(root, file)
            filename = Path(example_path).stem
            df = pd.read_csv(example_path)
            channel = filename[0]            
            if (training_dfs.get(channel) == None):
                training_dfs[channel] = []
                file_names["train"][channel] = []
                training_dfs[channel].append(df)
                file_names["train"][channel].append(filename)
            else:
                training_dfs[channel].append(df)
                file_names["train"][channel].append(filename)
    
    for root, _, files in os.walk(test_data_path):
        for file in files:
            example_path = os.path.join(root, file)
            filename = Path(example_path).stem
            df = pd.read_csv(example_path)
            channel = filename[0]
            if (test_dfs.get(channel) == None):
                test_dfs[channel] = []
                file_names["test"][channel] = []
                test_dfs[channel].append(df)
                file_names["test"][channel].append(filename)
            else:
                test_dfs[channel].append(df)
                file_names["test"][channel].append(filename)

    return (training_dfs, test_dfs, label_df, file_names)

def window_data(training_dfs, test_dfs, window_size, window_difference, label_df, file_names):
    """
    Apply sliding window segmentation to telemetry streams and generate labels.
    Windows are transposed to (features x time) and partial windows at the end of streams 
    are padded by taking the final window.
    Windows in test data are labeled as anomalous if they overlap with any annotated anomaly
    interval for that stream.

    Args:
        training_dfs (dict): Training telemetry grouped by channel.
        test_dfs (dict): Test telemetry grouped by channel.
        window_size (int): Number of timesteps per window.
        window_difference (int): Stride between consecutive windows.
        label_df (pd.DataFrame): DataFrame containing anomaly intervals.
        file_names (dict): Mapping of channel IDs to telemetry file identifiers.

    Returns:
        training_data (dict): Windowed training data per channel.
        test_data (dict): Windowed test data per channel.
        label_data (dict): Window-level anomaly labels per channel.
    """
    training_data = {}
    test_data = {}
    label_data = {}
    
    for channel in training_dfs.keys():
        training_data[channel] = []
        for df in training_dfs[channel]: 
            for i in range(0, len(df), window_difference):
                window = []
                if i + window_size > len(df): 
                    window = df.iloc[-window_size:].to_numpy().tolist()
                else: 
                    window = df.iloc[i:i + window_size].to_numpy().tolist()
                np_window = np.array(window)
                transposed_window = np_window.T
                normal_window = transposed_window.tolist() 
                training_data[channel].append(normal_window)
                
    for channel in test_dfs.keys():
        test_data[channel] = []
        label_data[channel] = []
        for i in range(len(test_dfs[channel])):
            df = test_dfs[channel][i]
            for j in range(0, len(df), window_difference):
                window = []
                if j + window_size > len(df): 
                    window = df.iloc[-window_size:].to_numpy().tolist()
                else: 
                    window = df.iloc[j:j + window_size].to_numpy().tolist()
                row_indices = label_df[label_df["chan_id"] == file_names["test"][channel][i]].index.tolist()
                
                if not row_indices:
                    continue

                anomaly_sequence = label_df.loc[row_indices[0], 'anomaly_sequences']
                anomaly_sequence = ast.literal_eval(anomaly_sequence)
                labeled = False
                for anomalies in anomaly_sequence:
                    if (not(anomalies[1] <= j or anomalies[0] >= j + window_size)) and labeled == False:
                        label_data[channel].append(1)
                        labeled = True
                if labeled == False:
                    label_data[channel].append(0)
                np_window = np.array(window)
                transposed_window = np_window.T
                normal_window = transposed_window.tolist() 
                test_data[channel].append(normal_window)
    print("[✓] Windowed raw data")     
    return (training_data, test_data, label_data)

def save_data(X_train_collection, X_test_collection, y_test_collection):
    """
    Save windowed datasets to disk in NumPy format under:
        data/processed/windowed/<channel>/{train,test,label}/

    Args:
        X_train_collection (dict): Windowed training data per channel.
        X_test_collection (dict): Windowed test data per channel.
        y_test_collection (dict): Window-level labels per channel.
    """
    data = {}
    os.makedirs("data/processed/windowed", exist_ok=True)

    for channel in X_train_collection:
        if data.get(channel) is None:
            data[channel] = {}
        if data.get(channel).get("train") is None:
            X_train = np.array(X_train_collection[channel])
            path = f"data/processed/windowed/{channel}/train"
            os.makedirs(path, exist_ok=True)
            np.save(path + f"/{channel}.npy", X_train)
            data[channel]["train"] = path + f"/{channel}.npy"
        else:
            print("Error: duplicate training data sets")
    
        
    for channel in X_test_collection: 
        if data.get(channel) is None:
            continue
        if data.get(channel).get("test") is None:
            X_test = np.array(X_test_collection[channel])
            path = f"data/processed/windowed/{channel}/test"
            os.makedirs(path, exist_ok=True)
            np.save(path + f"/{channel}.npy", X_test)
            data[channel]["test"] = path + f"/{channel}.npy"
        else:
            print("Error: duplicate testing data sets")
    
    for channel in y_test_collection:
        if data.get(channel) is None:
            continue
        if data.get(channel).get("label") is None:
            y_test = np.array(y_test_collection[channel])
            path = f"data/processed/windowed/{channel}/label"
            os.makedirs(path, exist_ok=True)
            np.save(path + f"/{channel}.npy", y_test)
            data[channel]["label"] = path + f"/{channel}.npy"
        else:
            print("Error: duplicate label data sets")
    print("[✓] Saved windowed data")     

def parse_args():
    """
    Parse command-line arguments for window configuration.

    Returns: argparse.Namespace: Parsed arguments containing window size and overlap.
    """

    parser = argparse.ArgumentParser(description="Initialize sliding windows for telemetry anomaly detection")
    parser.add_argument(
        "--size",
        type=int,
        default=60,
        help="Window size in timesteps (default: 60)"
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=40,
        help="Overlap between windows (default: 40)"
    )
    return parser.parse_args()

def main():
    """
    Execute the full preprocessing and windowing pipeline.

    Steps:
        1. Parse command-line arguments
        2. Load and convert raw telemetry data
        3. Apply sliding window segmentation
        4. Save processed datasets to disk
    """
    args = parse_args()
    load_data()
    window_size = args.size 
    window_difference = args.size - args.overlap 
    training_dfs, test_dfs, label_df, file_names = retreive_data()
    X_train_collection, X_test_collection, y_test_collection = window_data(training_dfs, test_dfs, window_size, window_difference, label_df, file_names)
    save_data(X_train_collection, X_test_collection, y_test_collection)

if __name__ == "__main__":
    main()
    