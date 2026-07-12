import copy
from collections import defaultdict
import subprocess
import yaml
import sys
import argparse
from src.utils.utils import UniversalTimer
from src.utils.logging_config import get_file_logger, setup_central_logging
from src.utils.settings import Settings

settings = Settings(_yaml_file="./config/runs/env_params.yaml")
setup_central_logging(log_file_path=f"reports/runs/{settings.name}/system_logs.log", global_labels={"project": "tuh", "environment": "dev"})

logger = get_file_logger(__name__, "main")
logger.info("Starting")


@UniversalTimer("Script Execution time")
def run_script(script_path, script_args_dict):
    """
    Runs a given Python script with specified arguments.
    Constructs command line arguments from the dictionary.
    """
    command = [sys.executable, script_path]  # sys.executable ensures using the correct python interpreter
    if script_args_dict:
        for arg_name, arg_value in script_args_dict.items():
            command.append(f"--{arg_name.replace('_', '-')}") # Convert snake_case to kebab-case for CLI
            command.append(str(arg_value))

    logger.info(f"\n[Orchestrator] Executing: {' '.join(command)}")
    try:
        # capture_output=True can be used to get stdout/stderr if needed
        # check=True will raise CalledProcessError if the script exits with a non-zero code
        result = subprocess.run(command, check=True)
        logger.info(f"[Orchestrator] Successfully executed {script_path}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"[Orchestrator] Error executing {script_path}.")
        logger.error(f"Return code: {e.returncode}")
        logger.error(f"STDOUT:\n{e.stdout.strip() if e.stdout else 'N/A'}")
        logger.error(f"STDERR:\n{e.stderr.strip() if e.stderr else 'N/A'}")
        return False
    except FileNotFoundError:
        logger.error(f"[Orchestrator] Error: Script not found at {script_path}")
        return False


@UniversalTimer("Pipeline Execution time")
def main():
    parser = argparse.ArgumentParser(description="Pipeline Orchestrator")
    parser.add_argument(
        "--config",
        type=str,
        default="config/orchestrator_config.yaml",
        help="Path to the YAML configuration file (default: config/orchestrator_config.yaml)",
    )
    args = parser.parse_args()

    with open(args.config, "r") as get_config_yaml:
        config_variables = yaml.safe_load(get_config_yaml)
        run_history = copy.deepcopy(config_variables)
        run_history["generated"] = defaultdict()

    pipeline_steps = run_history["scripts_to_run"]
    logger.info("[Orchestrator] Starting pipeline execution...")

    for i, step in enumerate(pipeline_steps):
        step_name = step.get("name", f"Step {i + 1}")
        script_path = step.get("script")
        script_args = step.get("args", {})

        logger.info(f"\n--- [Orchestrator] Running {step_name}: {script_path} ---")

        if not run_script(script_path, script_args):
            logger.error(f"[Orchestrator] Pipeline execution failed at {step_name}.")
            logger.error(f"[Orchestrator] Stopping pipeline execution.")
            break  # Stop pipeline on script failure
    else:  # This 'else' belongs to the 'for' loop, executes if the loop completed without 'break'
        logger.info("\n[Orchestrator] Pipeline execution completed successfully.")


if __name__ == "__main__":
    import time

    start_time = time.time()
    main()
    end_time = time.time()
    logger.info(f"\n[Orchestrator] Total pipeline execution time: {end_time - start_time:.2f} seconds")
