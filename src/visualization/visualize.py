import matplotlib

matplotlib.use("Agg")


import matplotlib.pyplot as plt
import pandas as pd


def plot_bar_chart(series: pd.Series, title: str, xlabel: str, ylabel: str, output_path: str):
    """
    Plots a simple bar chart from a pandas Series.
    """
    plt.figure(figsize=(10, 6))
    series.plot(kind="bar", color="skyblue", edgecolor="black")
    plt.title(title, fontsize=20)
    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    plt.xticks(rotation=45, ha="right", fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_stacked_bar_chart(df: pd.DataFrame, title: str, xlabel: str, ylabel: str, output_path: str):
    """
    Plots a stacked bar chart from a pandas DataFrame.
    """
    plt.figure(figsize=(12, 7))
    df.plot(kind="bar", stacked=True, colormap="viridis", edgecolor="black", ax=plt.gca())
    plt.title(title, fontsize=20)
    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    plt.xticks(rotation=45, ha="right", fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(title="Seizure", title_fontsize="13", fontsize="11")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
