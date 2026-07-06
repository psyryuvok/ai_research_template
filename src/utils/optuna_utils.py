import os
import shutil

import mlflow
import optuna
import optuna.visualization as vis
from codecarbon import OfflineEmissionsTracker


def final_cleanup(study: optuna.study.Study, base_model_dir: str):
    """
    Cleans up all non-best model trial folders from the base model directory at the end of the study.
    """
    print("\n[Final Cleanup] Starting final cleanup of non-best model trial folders...")
    best_model_path = study.user_attrs.get("best_model_path")
    best_model_trial_dir = None

    if best_model_path and os.path.exists(best_model_path):
        best_model_trial_dir = os.path.dirname(best_model_path)
        print(f"[Final Cleanup] Best model is in directory: {best_model_trial_dir}")
    else:
        print("[Final Cleanup] No best model path found or file doesn't exist.")

    folders_in_dir = 0
    removed_count = 0
    if not os.path.exists(base_model_dir):
        print(f"[Final Cleanup] Base model directory '{base_model_dir}' does not exist.")
        return

    for item_name in os.listdir(base_model_dir):
        item_path = os.path.join(base_model_dir, item_name)
        if os.path.isdir(item_path):
            folders_in_dir += 1
            if item_path != best_model_trial_dir:
                try:
                    shutil.rmtree(item_path)
                    print(f"[Final Cleanup] Removed non-best trial folder: {item_path}")
                    removed_count += 1
                except OSError as e:
                    print(f"[Final Cleanup] Error removing trial folder {item_path}: {e}")
            else:
                print(f"[Final Cleanup] Kept best model's trial folder: {item_path}")
    print(f"[Final Cleanup] Scanned {folders_in_dir} folders. Removed {removed_count} non-best trial folders.")


def callback_with_cleanup(study: optuna.study.Study, trial: optuna.trial.FrozenTrial):
    """
    Callback executed after each trial.
    If the current trial is the new best, it updates the study's best model path
    and removes the folder of the previously saved best model.
    """
    if study.best_trial is not None and study.best_trial.number == trial.number:
        new_best_model_path = trial.user_attrs.get("model_path")

        if not new_best_model_path:
            print(f"[Callback] Warning: model_path not found in user_attrs for new best trial {trial.number}.")
            return

        old_best_model_path = study.user_attrs.get("best_model_path")

        if old_best_model_path and old_best_model_path != new_best_model_path:
            old_model_dir = os.path.dirname(old_best_model_path)
            try:
                if os.path.exists(old_model_dir):
                    shutil.rmtree(old_model_dir)
                    print(f"[Callback] Removed previous best model's directory: {old_model_dir}")
                else:
                    print(f"[Callback] Info: Previous best model directory not found for removal: {old_model_dir}")
            except OSError as e:
                print(f"[Callback] Error removing previous best model directory {old_model_dir}: {e}")

        study.set_user_attr(key="best_model_path", value=new_best_model_path)
        print(f"[Callback] New best trial: {trial.number}. Best model path updated to: {new_best_model_path}")


def log_optuna_plots(study: optuna.study.Study):
    """Logs Optuna visualizations to MLflow."""
    try:
        optimization_history_plot = vis.plot_optimization_history(study)
        mlflow.log_figure(optimization_history_plot, "optimization_history_plot.html")
    except Exception as e:
        print(f"Could not log optimization history plot: {e}")

    try:
        parallel_coordinate_plot = vis.plot_parallel_coordinate(study)
        mlflow.log_figure(parallel_coordinate_plot, "parallel_coordinate_plot.html")
    except Exception as e:
        pass  # Fails if < 2 trials or certain conditions

    try:
        param_importance_plot = vis.plot_param_importances(study)
        mlflow.log_figure(param_importance_plot, "param_importance_plot.html")
    except Exception as e:
        pass


def setup_tracker(project_name: str, output_dir: str, output_file: str = "emissions_log.csv"):
    """Creates and starts an OfflineEmissionsTracker."""
    os.makedirs(output_dir, exist_ok=True)
    tracker = OfflineEmissionsTracker(
        project_name=project_name,
        output_dir=output_dir,
        output_file=output_file,
        log_level="warning",
        country_iso_code="ROU",  # Adjust based on location for more accurate emissions tracking
    )
    tracker.start()
    return tracker
