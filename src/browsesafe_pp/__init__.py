"""BrowseSafe++: paired prompt-injection detection data, evaluation and adaptive attacks."""
from .dataset import (
    ALL_VIEWS, DEFAULT_DATASET, DEFAULT_REVISION, PRIVATE_SPLITS, SCREENSHOT_VIEW, VIEWS, Benchmark,
    DetectorInput, Page, Pair, Prediction, dataset_id, describe, iter_batches, load,
)
from .evaluation.classification import DetectionReport, ModelDetector, evaluate, evaluate_async
from .providers import OpenAICompatible, ProviderError
from . import adaptive, evaluation, providers

__version__ = "0.1.0"

__all__ = [
    "ALL_VIEWS", "DEFAULT_DATASET", "DEFAULT_REVISION", "PRIVATE_SPLITS", "SCREENSHOT_VIEW", "VIEWS", "Benchmark",
    "DetectionReport", "DetectorInput", "ModelDetector", "OpenAICompatible", "Page", "Pair", "Prediction",
    "ProviderError", "adaptive", "dataset_id", "describe", "evaluate", "evaluate_async", "evaluation",
    "iter_batches", "load", "providers",
]
