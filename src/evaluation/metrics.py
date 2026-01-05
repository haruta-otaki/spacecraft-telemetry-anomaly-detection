from pathlib import Path
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_results():
    """
    Load evaluation results for models from CSV files.

    Returns
    results (dict): Nested dictionary containing scores of various metrics for each model and channel.
    scores (list): Sorted list of metric names found across all models.
    """
    results = {}
    scores = set()
    results_path = "results/data_frame_format"

    for file in os.listdir(results_path):
        model_name = Path(file).stem
        df = pd.read_csv(os.path.join(results_path, file), index_col=0)

        results[model_name] = df.to_dict(orient="index")
        scores.update(df.columns)

    return results, sorted(scores)

def create_plot(results, score):
    """
    Create a grouped bar plot for a single evaluation metric.

    Parameters
    results (dict): Nested dictionary containing scores of various metrics for each model and channel.

    score (str): Name of the metric to plot.
    """
    models = list(results.keys())
    channels = list(results.values())[0].keys()

    bar_width = 0.8 / len(models)
    x = np.arange(len(channels))

    plt.figure(figsize=(8, 4))
    plt.title(score)

    index = 0
    for model in models:

        values = []
        for channel in channels: 
            values.append(results[model][channel][score])

        plt.bar(x + index * bar_width, values, width=bar_width, label=model)
        index += 1

    plt.xticks(x + bar_width * (len(models) - 1) / 2, channels)
    plt.xlabel("Channel")
    plt.ylabel(score)
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"results/figures/{score}.png")

def main():
    """
    Loads results for all models and generates one bar plot per evaluation metric.
    """
    results, scores = load_results()
    for score in scores: 
     create_plot(results, score)

if __name__ == "__main__":
    main()