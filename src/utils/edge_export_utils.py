import logging
import os
import subprocess
import tempfile

import tensorflow as tf

logger = logging.getLogger("edge_export")


def export_tflite(model, output_path, quantization="none", representative_dataset_gen=None):
    """
    Exports a Keras model to TFLite format.

    :param model: The Keras model to export.
    :param output_path: Path to save the .tflite file.
    :param quantization: 'none', 'fp16', or 'int8'.
    :param representative_dataset_gen: A generator function yielding dictionaries of inputs for INT8 calibration.
    """
    logger.info(f"Exporting TFLite model to {output_path} (Quantization: {quantization})")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    if quantization == "fp16":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    elif quantization == "int8":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        if representative_dataset_gen:
            logger.info("Setting up representative dataset for INT8 calibration...")
            converter.representative_dataset = representative_dataset_gen
            # Ensure full integer quantization
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
        else:
            logger.warning("INT8 quantization requested but no calibration generator provided. Using dynamic range quantization instead.")

    try:
        tflite_model = converter.convert()
        with open(output_path, "wb") as f:
            f.write(tflite_model)
        logger.info("TFLite export successful.")
    except Exception as e:
        logger.error(f"Failed to export TFLite: {e}")


def export_onnx(model, output_path):
    """
    Exports a Keras model to ONNX format using tf2onnx.

    :param model: The Keras model to export.
    :param output_path: Path to save the .onnx file.
    """
    logger.info(f"Exporting ONNX model to {output_path}")
    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            saved_model_path = os.path.join(tmp_dir, "saved_model")

            # Export to SavedModel format (Keras 3 syntax, fallback to Keras 2)
            try:
                model.export(saved_model_path)
            except AttributeError:
                tf.saved_model.save(model, saved_model_path)

            logger.info("Converting SavedModel to ONNX using tf2onnx CLI...")
            result = subprocess.run(
                ["python", "-m", "tf2onnx.convert", "--saved-model", saved_model_path, "--output", output_path, "--opset", "13"],
                capture_output=True,
                text=True,
            )

            if result.returncode == 0:
                logger.info("ONNX export successful.")
            else:
                logger.error(f"Failed to export ONNX: {result.stderr}")
    except Exception as e:
        logger.error(f"Failed to export ONNX: {e}")
