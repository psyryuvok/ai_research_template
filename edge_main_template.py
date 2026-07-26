import argparse
import logging
import time

import numpy as np

# import YOUR_PREPROCESSOR_HERE
from src.models.edge_inference import EdgeModelPredictor
from src.utils.logging_config import get_file_logger, setup_central_logging
from src.utils.settings import get_settings
from src.utils.utils import UniversalTimer

settings = get_settings()
logger = get_file_logger(__name__, "edge")


def generate_dummy_data():
    """
    Template function: Replace this with your specific dummy data generation
    logic for edge simulation (e.g., loading an image, random text, etc.)
    """
    logger.info("Simulating dummy raw data...")
    # Example for Image Model:
    # return np.random.randn(224, 224, 3).astype(np.float32)
    return {"input_1": np.random.randn(10, 10).astype(np.float32)}


@UniversalTimer("main")
def main(model_path: str, model_type: str):
    try:
        predictor = EdgeModelPredictor(model_path=model_path, model_type=model_type)
        logger.info(f"Model ({model_type}) loaded successfully from {model_path}.")
    except Exception as e:
        logger.warning(f"Could not load model at {model_path}. Make sure to export it first! Error: {e}")
        return

    # TODO: Instantiate your concrete Preprocessor class here
    # preprocessor = YourCustomPreprocessor(...)

    raw_data = generate_dummy_data()

    start_time = time.time()
    # TODO: Replace with your concrete preprocessor call
    # windows = preprocessor.prepare_inference_data(raw_data)
    windows = [raw_data]  # Mocking preprocessor output for the template
    prep_time = time.time() - start_time
    logger.info(f"Preprocessing completed in {prep_time:.4f}s. Generated {len(windows)} windows.")

    for i, window in enumerate(windows):
        start_time = time.time()

        predictions = predictor.predict(window)
        inf_time = time.time() - start_time

        logger.info(f"Window {i + 1} Inference Time: {inf_time:.4f}s")
        for out_name, out_val in predictions.items():
            logger.info(f"  -> {out_name}: {out_val.flatten()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Edge Inference Script")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model file")
    parser.add_argument("--model_type", type=str, choices=["tflite", "onnx"], default="tflite", help="Type of model (tflite or onnx)")
    args = parser.parse_args()
    setup_central_logging(log_file_path=f"reports/runs/{settings.name}/system_logs.log")
    main(model_path=args.model_path, model_type=args.model_type)
