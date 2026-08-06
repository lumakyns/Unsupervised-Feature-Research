"""Hardcoded dataset definitions for separation models."""

from dataclasses import dataclass
from enum import Enum


class Dataset(str, Enum):
    """Datasets supported by models in this experiment."""

    CIFAR10 = "cifar10"
    MNIST = "mnist"


@dataclass(frozen=True)
class DatasetSpec:
    """Model-facing input and target dimensions."""

    channels: int
    height: int
    width: int
    num_classes: int


DATASET_SPECS: dict[Dataset, DatasetSpec] = {
    Dataset.CIFAR10: DatasetSpec(channels=3, height=32, width=32, num_classes=10),
    Dataset.MNIST: DatasetSpec(channels=1, height=28, width=28, num_classes=10),
}


def get_dataset_spec(dataset: Dataset) -> DatasetSpec:
    """Return initialization metadata for a supported dataset."""

    try:
        return DATASET_SPECS[dataset]
    except KeyError as error:
        raise ValueError(f"Unsupported dataset: {dataset!r}") from error
