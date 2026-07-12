import abc
import os

import mlflow
import optuna
import tensorflow as tf
from optuna.integration.mlflow import MLflowCallback
from optuna.integration.tensorboard import TensorBoardCallback

from src.utils.optuna_utils import callback_with_cleanup, final_cleanup, log_optuna_plots
from src.utils.utils import disable_randomness, get_or_create_mflow_experiment, setup_tracker


class BaseOptunaKerasPipeline(abc.ABC):
    """
    Generic template for Keras + Optuna + MLflow training pipelines.
    Supports Multi-Input and Multi-Head inherently via tf.keras.
    """

    def __init__(self, config, study_name: str, metric_to_track: str, direction: str = "maximize"):
        self.config = config
        self.study_name = study_name
        self.metric_to_track = metric_to_track
        self.direction = direction

        disable_randomness(self.config.repeatability)
        self.experiment_id = get_or_create_mflow_experiment(self.config.name)
        mlflow.set_experiment(experiment_id=self.experiment_id)

        self.runs_dir = f"./reports/runs/{self.config.name}/{self.study_name}"
        self.base_model_dir = f"./models/experimenting/{self.config.name}/{self.study_name}"
        os.makedirs(self.runs_dir, exist_ok=True)
        os.makedirs(self.base_model_dir, exist_ok=True)

    @abc.abstractmethod
    def get_datasets(self):
        """Should return (train_dataset, dev_dataset, eval_dataset, metadata...)"""
        pass

    @abc.abstractmethod
    def build_and_compile_model(self, trial: optuna.Trial, metadata):
        """Should return a compiled tf.keras.Model and the suggested batch_size"""
        pass

    @abc.abstractmethod
    def evaluate_trial(self, model, train_dataset_batched, dev_dataset_batched, eval_dataset_batched, metadata, trial_number, trial_specific_dir):
        """Custom evaluation hook (e.g., Confusion Matrices, custom metrics)"""
        pass

    def objective(self, trial: optuna.Trial, train_dataset, dev_dataset, eval_dataset, metadata) -> float:
        trial_tracker = setup_tracker(project_name=f"trial_{trial.number}", output_dir=self.runs_dir, output_file=f"T-{trial.number}-emissions_log.csv")
        tf.keras.backend.clear_session()

        # 1. Build Model
        model, batch_size = self.build_and_compile_model(trial, metadata)

        train_batched = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        dev_batched = dev_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

        def remove_metadata(*args):
            inputs = args[0]
            inputs_clean = {k: v for k, v in inputs.items() if not k.startswith("meta_info_")}
            if len(args) == 3:
                return inputs_clean, args[1], args[2]
            return inputs_clean, args[1]

        train_fit = train_batched.map(remove_metadata, num_parallel_calls=tf.data.AUTOTUNE)
        dev_fit = dev_batched.map(remove_metadata, num_parallel_calls=tf.data.AUTOTUNE)

        # 2. Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor=self.metric_to_track, patience=20, mode="max", restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor=self.metric_to_track, factor=0.5, patience=10, mode="max", min_lr=1e-8),
        ]

        # 3. Train
        model.fit(train_fit, validation_data=dev_fit, epochs=self.config.model_seizure.epochs, callbacks=callbacks, verbose=1)

        # 4. Save Trial Model
        trial_specific_dir = os.path.join(self.base_model_dir, f"T-{trial.number}")
        os.makedirs(trial_specific_dir, exist_ok=True)
        model_path = os.path.join(trial_specific_dir, "tf_model.keras")
        model.save(model_path)
        trial.set_user_attr(key="model_path", value=model_path)

        # 5. Evaluate and Return Metric
        eval_batched = eval_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        self.evaluate_trial(model, train_batched, dev_batched, eval_batched, metadata, trial.number, trial_specific_dir)

        results = model.evaluate(dev_fit, return_dict=True, verbose=1)

        emissions = trial_tracker.stop()
        if emissions is not None:
            mlflow.log_metric("emissions_kg_CO2", emissions)

        # NOTE - Prioritize domain specific metric, fallback to general tracking metric
        if "seizure_type_f1_score" in results:
            return results["seizure_type_f1_score"]
        elif "f1_score" in results:
            return results["f1_score"]
        return results.get(self.metric_to_track.replace("val_", ""), 0.0)

    def run_optimization(self, n_trials: int):
        train_dataset, dev_dataset, eval_dataset, metadata = self.get_datasets()

        mlflow_callback = MLflowCallback(
            tracking_uri=mlflow.get_tracking_uri(), create_experiment=False, metric_name=self.metric_to_track, mlflow_kwargs={"nested": True}
        )

        global_tracker = setup_tracker(project_name=self.study_name, output_dir=self.runs_dir, output_file="emissions_log.csv")

        with mlflow.start_run(experiment_id=self.experiment_id, run_name=self.study_name):
            mlflow.tensorflow.autolog(checkpoint=True, keras_model_kwargs={"save_format": "keras"}, saved_model_kwargs={"save_format": "keras"})

            study = optuna.create_study(
                study_name=self.study_name,
                direction=self.direction,
                load_if_exists=True,
                storage=f"sqlite:///{self.runs_dir}/optuna.db",
                sampler=optuna.samplers.TPESampler(seed=self.config.repeatability.seed),
            )

            tb_callback = TensorBoardCallback(f"./reports/tensorboard/{self.config.name}/logs_optuna/{self.study_name}", metric_name=self.metric_to_track)

            @mlflow_callback.track_in_mlflow()
            def objective_wrapper(t):
                return self.objective(t, train_dataset, dev_dataset, eval_dataset, metadata)

            study.optimize(objective_wrapper, n_trials=n_trials, callbacks=[callback_with_cleanup, tb_callback, mlflow_callback], gc_after_trial=True)

            # Save the very best model to a clear path as well
            best_model_path_from_study = study.user_attrs.get("best_model_path")
            if best_model_path_from_study and os.path.exists(best_model_path_from_study):
                loaded_best_model = tf.keras.models.load_model(best_model_path_from_study, compile=False)
                # Ensure the overall best model is saved
                loaded_best_model.save(f"./reports/runs/{self.config.name}/{self.study_name}/pipeline_model_best_{self.study_name}.keras")

                best_batch_size = study.best_trial.params.get("optuna_batch_size", 512)
                train_batched_best = train_dataset.batch(best_batch_size).prefetch(tf.data.AUTOTUNE)
                dev_batched_best = dev_dataset.batch(best_batch_size).prefetch(tf.data.AUTOTUNE)
                eval_batched_best = eval_dataset.batch(best_batch_size).prefetch(tf.data.AUTOTUNE)

                self.evaluate_trial(loaded_best_model, train_batched_best, dev_batched_best, eval_batched_best, metadata, None, self.base_model_dir)

            final_cleanup(study, self.base_model_dir)
            log_optuna_plots(study)

            total_emissions = global_tracker.stop()
            if total_emissions:
                mlflow.log_metric("total_study_emissions_kg_CO2", total_emissions)

        return study.best_trial
