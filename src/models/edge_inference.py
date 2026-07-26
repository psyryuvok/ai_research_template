import logging
from typing import Any

import numpy as np

logger = logging.getLogger("edge_inference")
logger.setLevel(logging.INFO)


class EdgeModelPredictor:
    """
    A lightweight inference wrapper designed to run on Edge devices (like Raspberry Pi).
    It avoids importing heavy frameworks like tensorflow directly.
    Instead, it conditionally uses tflite_runtime or onnxruntime.
    """

    def __init__(self, model_path: str, model_type: str = "tflite"):
        """
        :param model_path: Path to the .tflite or .onnx file.
        :param model_type: Either "tflite" or "onnx"
        """
        self.model_path = model_path
        self.model_type = model_type.lower()
        self._interpreter = None
        self._onnx_session = None
        self.input_details = None
        self.output_details = None

        if self.model_type == "tflite":
            self._init_tflite()
        elif self.model_type == "onnx":
            self._init_onnx()
        else:
            raise ValueError("model_type must be 'tflite' or 'onnx'")

    def _init_tflite(self):
        try:
            # Prefer LiteRT (formerly TFLite) on edge to save memory
            import ai_edge_litert.interpreter as tflite  # type: ignore[import-not-found]
        except ImportError:
            try:
                import tflite_runtime.interpreter as tflite  # type: ignore[import-not-found]
            except ImportError:
                # Fallback for desktop testing
                import tensorflow as tf

                tflite = tf.lite

        logger.info(f"Loading TFLite model from {self.model_path}")
        self._interpreter = tflite.Interpreter(model_path=self.model_path)
        self._interpreter.allocate_tensors()

        self.input_details = self._interpreter.get_input_details()
        self.output_details = self._interpreter.get_output_details()

        # Build mapping from tensor name to index for easy feeding
        self.input_name_to_index = {det["name"].split(":")[0]: det["index"] for det in self.input_details}

    def _init_onnx(self):
        import onnxruntime as ort  # type: ignore[import-untyped]

        logger.info(f"Loading ONNX model from {self.model_path}")
        self._onnx_session = ort.InferenceSession(self.model_path)

        self.input_details = self._onnx_session.get_inputs()
        self.output_details = self._onnx_session.get_outputs()

        self.input_names = [inp.name for inp in self.input_details]
        self.output_names = [out.name for out in self.output_details]

    def _prepare_inputs_tflite(self, inputs):
        """Matches inputs to model input tensors"""
        if isinstance(inputs, np.ndarray):
            inputs = [inputs]

        if isinstance(inputs, (list, tuple)):
            if len(inputs) != len(self.input_details):
                logger.warning(f"Warning: Model expects {len(self.input_details)} inputs but {len(inputs)} were provided.")

            for i, tensor_data in enumerate(inputs):
                if i >= len(self.input_details):
                    break

                tensor_idx = self.input_details[i]["index"]
                # Add batch dimension if missing
                shape_len_diff = len(self.input_details[i]["shape"]) - len(np.shape(tensor_data))
                data = np.array([tensor_data]) if shape_len_diff == 1 else np.array(tensor_data)

                # Check quantization specifics
                target_dtype = self.input_details[i].get("dtype", np.float32)
                if target_dtype == np.int8:
                    scale, zero_point = self.input_details[i]["quantization"]
                    if scale > 0.0:
                        data = data / scale + zero_point
                    data = data.astype(np.int8)
                else:
                    data = data.astype(target_dtype)

                self._interpreter.set_tensor(tensor_idx, data)

        elif isinstance(inputs, dict):
            used_keys = set()
            for input_name, tensor_idx in self.input_name_to_index.items():
                # Extract base name to match dictionary keys (e.g., 'input_FP1-F7')
                base_name = input_name.split(";")[0].strip()

                # Find matching key in inputs
                matched_key = None
                for key in inputs.keys():
                    if key == base_name or key in base_name:
                        matched_key = key
                        break

                if matched_key:
                    used_keys.add(matched_key)
                    shape_len_diff = len(self.input_details[tensor_idx]["shape"]) - len(np.shape(inputs[matched_key]))
                    data = np.array([inputs[matched_key]]) if shape_len_diff == 1 else np.array(inputs[matched_key])
                else:
                    logger.warning(f"Warning: Input tensor {base_name} not provided in inputs. Using zeros.")
                    shape = self.input_details[tensor_idx]["shape"]
                    shape = [dim if dim > 0 else 1 for dim in shape]
                    data = np.zeros(shape)

                # Check quantization specifics
                target_dtype = self.input_details[tensor_idx].get("dtype", np.float32)
                if target_dtype == np.int8:
                    scale, zero_point = self.input_details[tensor_idx]["quantization"]
                    if scale > 0.0:
                        data = data / scale + zero_point
                    data = data.astype(np.int8)
                else:
                    data = data.astype(target_dtype)

                self._interpreter.set_tensor(tensor_idx, data)

            unused_keys = set(inputs.keys()) - used_keys
            if unused_keys:
                logger.warning(f"Warning: The following keys in inputs were not used by the TFLite model: {', '.join(unused_keys)}")
        else:
            raise ValueError("Inputs must be a dict, list, tuple, or numpy array.")

    def _prepare_inputs_onnx(self, inputs):
        ort_inputs = {}

        if isinstance(inputs, np.ndarray):
            inputs = [inputs]

        if isinstance(inputs, (list, tuple)):
            if len(inputs) != len(self.input_details):
                logger.warning(f"Warning: Model expects {len(self.input_details)} inputs but {len(inputs)} were provided.")

            for i, tensor_data in enumerate(inputs):
                if i >= len(self.input_details):
                    break
                inp_name = self.input_details[i].name
                # Add batch dim if missing
                shape_len_diff = len(self.input_details[i].shape) - len(np.shape(tensor_data))
                data = np.array([tensor_data]) if shape_len_diff == 1 else np.array(tensor_data)
                ort_inputs[inp_name] = data.astype(np.float32)

        elif isinstance(inputs, dict):
            used_keys = set()
            for inp_detail in self.input_details:
                inp_name = inp_detail.name
                matched_key = None
                for key in inputs.keys():
                    if key == inp_name or key in inp_name:
                        matched_key = key
                        break

                if matched_key:
                    used_keys.add(matched_key)
                    shape_len_diff = len(inp_detail.shape) - len(np.shape(inputs[matched_key]))
                    data = np.array([inputs[matched_key]]) if shape_len_diff == 1 else np.array(inputs[matched_key])
                    ort_inputs[inp_name] = data.astype(np.float32)
                else:
                    logger.warning(f"Warning: Input tensor {inp_name} not provided in inputs. Using zeros.")
                    shape = [dim if isinstance(dim, int) else 1 for dim in inp_detail.shape]
                    ort_inputs[inp_name] = np.zeros(shape, dtype=np.float32)

            unused_keys = set(inputs.keys()) - used_keys
            if unused_keys:
                logger.warning(f"Warning: The following keys in inputs were not used by the ONNX model: {', '.join(unused_keys)}")
        else:
            raise ValueError("Inputs must be a dict, list, tuple, or numpy array.")

        return ort_inputs

    def predict(self, inputs: Any) -> dict:
        """
        Runs inference on a single window/input.
        :param inputs: Numpy array, list of numpy arrays, or Dictionary mapping string keys to numpy arrays.
                       Batch dimension will be added automatically if missing.
        :return: Dictionary of predictions.
        """
        if self.model_type == "tflite":
            assert self._interpreter is not None
            assert self.output_details is not None
            self._prepare_inputs_tflite(inputs)
            self._interpreter.invoke()

            results = {}
            for out in self.output_details:
                data = self._interpreter.get_tensor(out["index"])

                # Dequantize if int8
                if out["dtype"] == np.int8:
                    scale, zero_point = out["quantization"]
                    data = (data.astype(np.float32) - zero_point) * scale

                results[out["name"]] = data
            return results

        elif self.model_type == "onnx":
            assert self._onnx_session is not None
            ort_inputs = self._prepare_inputs_onnx(inputs)
            ort_outs = self._onnx_session.run(None, ort_inputs)

            return {name: val for name, val in zip(self.output_names, ort_outs)}

        raise ValueError(f"Unknown model_type: {self.model_type}")
