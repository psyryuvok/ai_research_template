import os
import shutil

import mlflow
import optuna
import optuna.visualization as vis
from optuna.importance import FanovaImportanceEvaluator, MeanDecreaseImpurityImportanceEvaluator, PedAnovaImportanceEvaluator

from src.utils.logging_config import get_file_logger
from src.utils.settings import Settings

settings = Settings()
logger = get_file_logger(__name__, "optuna")


def final_cleanup(study: optuna.study.Study, base_model_dir: str):
    """
    Cleans up all non-best model trial folders from the base model directory at the end of the study.
    """
    logger.info("\n[Final Cleanup] Starting final cleanup of non-best model trial folders...")
    best_model_path = study.user_attrs.get("best_model_path")
    best_model_trial_dir = None

    if best_model_path and os.path.exists(best_model_path):
        best_model_trial_dir = os.path.dirname(best_model_path)
        logger.info(f"[Final Cleanup] Best model is in directory: {best_model_trial_dir}")
    else:
        logger.info("[Final Cleanup] No best model path found or file doesn't exist.")

    folders_in_dir = 0
    removed_count = 0
    if not os.path.exists(base_model_dir):
        logger.info(f"[Final Cleanup] Base model directory '{base_model_dir}' does not exist.")
        return

    for item_name in os.listdir(base_model_dir):
        item_path = os.path.join(base_model_dir, item_name)
        if os.path.isdir(item_path):
            folders_in_dir += 1
            if item_path != best_model_trial_dir:
                try:
                    shutil.rmtree(item_path)
                    logger.info(f"[Final Cleanup] Removed non-best trial folder: {item_path}")
                    removed_count += 1
                except OSError as e:
                    logger.error(f"[Final Cleanup] Error removing trial folder {item_path}: {e}")
            else:
                logger.info(f"[Final Cleanup] Kept best model's trial folder: {item_path}")
    logger.info(f"[Final Cleanup] Scanned {folders_in_dir} folders. Removed {removed_count} non-best trial folders.")


def callback_with_cleanup(study: optuna.study.Study, trial: optuna.trial.FrozenTrial):
    """
    Callback executed after each trial.
    If the current trial is the new best, it updates the study's best model path
    and removes the folder of the previously saved best model.
    """
    if study.best_trial is not None and study.best_trial.number == trial.number:
        new_best_model_path = trial.user_attrs.get("model_path")

        if not new_best_model_path:
            logger.warning(f"[Callback] Warning: model_path not found in user_attrs for new best trial {trial.number}.")
            return

        old_best_model_path = study.user_attrs.get("best_model_path")

        if old_best_model_path and old_best_model_path != new_best_model_path:
            old_model_dir = os.path.dirname(old_best_model_path)
            try:
                if os.path.exists(old_model_dir):
                    shutil.rmtree(old_model_dir)
                    logger.info(f"[Callback] Removed previous best model's directory: {old_model_dir}")
                else:
                    logger.info(f"[Callback] Info: Previous best model directory not found for removal: {old_model_dir}")
            except OSError as e:
                logger.error(f"[Callback] Error removing previous best model directory {old_model_dir}: {e}")

        study.set_user_attr(key="best_model_path", value=new_best_model_path)
        logger.info(f"[Callback] New best trial: {trial.number}. Best model path updated to: {new_best_model_path}")


def log_optuna_plots(study: optuna.study.Study):
    """Logs Optuna visualizations to MLflow."""
    try:
        optimization_history_plot = vis.plot_optimization_history(study)
        mlflow.log_figure(optimization_history_plot, "optimization_history_plot.html")
    except Exception as e:
        logger.error(f"Could not log optimization history plot: {e}")

    try:
        parallel_coordinate_plot = vis.plot_parallel_coordinate(study)
        mlflow.log_figure(parallel_coordinate_plot, "parallel_coordinate_plot.html")
    except Exception as e:
        logger.error(f"Could not log parallel coordinate plot: {e}")

    try:
        # Default fANOVA plot
        param_importance_plot = vis.plot_param_importances(study)
        mlflow.log_figure(param_importance_plot, "param_importance_plot.html")

        fanova = FanovaImportanceEvaluator(seed=settings.repeatability.seed)
        fanova_plot = vis.plot_param_importances(study, evaluator=fanova)
        mlflow.log_figure(fanova_plot, "param_importance_plot_FanovaImportanceEvaluator.html")

        for quantile in [0.1, 0.25, 0.4]:
            ped_anova = PedAnovaImportanceEvaluator(baseline_quantile=quantile)
            if study.direction == optuna.study.StudyDirection.MAXIMIZE:
                ped_anova_plot = vis.plot_param_importances(study, evaluator=ped_anova, target=lambda t: float(t.value) if t.value is not None else 0.0)
            else:
                ped_anova_plot = vis.plot_param_importances(study, evaluator=ped_anova)

            mlflow.log_figure(ped_anova_plot, f"param_importance_plot_PedAnovaImportanceEvaluator_{quantile}.html")

        mdi = MeanDecreaseImpurityImportanceEvaluator(seed=settings.repeatability.seed)
        mdi_plot = vis.plot_param_importances(study, evaluator=mdi)
        mlflow.log_figure(mdi_plot, "param_importance_plot_MeanDecreaseImpurityImportanceEvaluator.html")

    except Exception as e:
        logger.error(f"Could not log parameter importance plot: {e}")
