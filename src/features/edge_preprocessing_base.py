import abc
from typing import Any, List


class AbstractEdgePreprocessor(abc.ABC):
    """
    Abstract base class for edge-friendly preprocessing modules.
    Project-specific preprocessing pipelines should inherit from this class.
    """

    @abc.abstractmethod
    def prepare_inference_data(self, raw_data: Any, *args, **kwargs) -> List[Any]:
        """
        End-to-end preparation of raw data for model inference.

        :param raw_data: The incoming raw data (e.g., dict of arrays, image array, text string).
        :return: A list of prepared data elements (e.g., windows, batches) suitable for prediction.
        """
        pass
