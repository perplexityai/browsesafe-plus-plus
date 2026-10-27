"""BrowseSafe++: paired prompt-injection detection data, evaluation and adaptive attacks."""
from .dataset import (
    ALL_VIEWS, DEFAULT_DATASET, DEFAULT_REVISION, SCREENSHOT_VIEW, VIEWS, Benchmark, DetectorInput,
    Page, Pair, Prediction, dataset_id, describe, iter_batches, load,
)
from .evaluation.classification import DetectionReport, ModelDetector, evaluate, evaluate_async
from .providers import OpenAICompatible, ProviderError
from .test_inputs import (
    LEADERBOARD_SPACE, TEST_DATASET, TestPage, load_test_inputs, predict_test, predict_test_async,
    write_predictions,
)
from . import adaptive, evaluation, providers

__version__ = "0.2.0"

__all__ = [
    "ALL_VIEWS", "DEFAULT_DATASET", "DEFAULT_REVISION", "LEADERBOARD_SPACE", "SCREENSHOT_VIEW", "TEST_DATASET",
    "VIEWS", "Benchmark", "DetectionReport", "DetectorInput", "ModelDetector", "OpenAICompatible", "Page", "Pair",
    "Prediction", "ProviderError", "TestPage", "adaptive", "dataset_id", "describe", "evaluate", "evaluate_async",
    "evaluation", "iter_batches", "load", "load_test_inputs", "predict_test", "predict_test_async", "providers",
    "write_predictions",
]
