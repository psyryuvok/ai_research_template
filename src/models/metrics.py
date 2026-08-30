import os

import matplotlib
import matplotlib.pyplot as plt
import mlflow

matplotlib.use("Agg")
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


def evaluate_multi_head_classification(
    model, dataset, dataset_name, active_targets, class_names_dict, num_classes_dict, figures_dir, stats_dir, trial_number=None, metadata_names=None
):
    """
    A generic template function to evaluate multi-head classification models.
    It iterates over multiple targets and generates absolute and normalized confusion matrices.
    """
    y_true_dict = {t: [] for t in active_targets}
    y_pred_dict = {t: [] for t in active_targets}

    for batch in dataset:
        inputs, labels = batch[0], batch[1]
        preds = model(inputs, training=False)

        # Standardize predictions to always be a dictionary (multi-head format)
        if not isinstance(preds, dict):
            if len(active_targets) == 1:
                preds = {active_targets[0]: preds}
            elif isinstance(preds, list):
                preds = {t: preds[i] for i, t in enumerate(active_targets)}

        for t in active_targets:
            y_true_dict[t].append(labels[t].numpy())
            y_pred_dict[t].append(preds[t].numpy())

    # Increase font size for plots globally for this context
    plt.rcParams.update({"font.size": 18})
    artifact_path = f"evaluation/trial_{trial_number}/{dataset_name}" if trial_number is not None else f"evaluation/best_model/{dataset_name}"

    for t in active_targets:
        if not y_true_dict[t]:
            continue

        y_true = np.argmax(np.concatenate(y_true_dict[t], axis=0), axis=-1)
        y_pred = np.argmax(np.concatenate(y_pred_dict[t], axis=0), axis=-1)

        display_labels = class_names_dict.get(t)
        labels_range = list(range(num_classes_dict.get(t, len(display_labels) if display_labels else 0)))

        _plot_and_save_cm(y_true, y_pred, labels_range, display_labels, dataset_name, t, figures_dir, artifact_path, normalize=None)
        _plot_and_save_cm(y_true, y_pred, labels_range, display_labels, dataset_name, t, figures_dir, artifact_path, normalize="true")


def _plot_and_save_cm(y_true, y_pred, labels_range, display_labels, dataset_name, target, save_dir, artifact_path, normalize):
    """Helper to plot and save a single confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, labels=labels_range, normalize=normalize)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)

    fig, ax = plt.subplots(figsize=(10, 10))
    values_format = ".2f" if normalize else None
    disp.plot(ax=ax, cmap="Blues", xticks_rotation="vertical", values_format=values_format, im_kw={"vmin": 0, "vmax": 1} if normalize else None)

    title_prefix = "Normalized " if normalize else ""
    ax.set_title(f"{title_prefix}Confusion Matrix {dataset_name}: {target}", fontsize=20)

    filename = f"confusion_matrix_{'normalized_' if normalize else ''}{dataset_name}_{target}.png"
    cm_path = os.path.join(save_dir, filename)
    plt.savefig(cm_path)
    plt.close(fig)
    mlflow.log_artifact(cm_path, artifact_path=artifact_path)
