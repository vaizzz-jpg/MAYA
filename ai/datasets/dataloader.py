"""DataLoader factory utilities for MAYA datasets."""

from __future__ import annotations

import logging
from typing import Any, Callable, Literal

from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

from ai.datasets.dataset import MayaImageDataset, build_split_dataset
from ai.datasets.dataset_config import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    DatasetConfig,
    SplitName,
    get_dataset_config,
)

logger = logging.getLogger("maya.datasets.dataloader")

TransformName = Literal["none", "tensor", "train", "eval"]


def build_transforms(
    name: TransformName,
    *,
    image_size: int = 224,
) -> Callable[[Image.Image], Any] | None:
    """Return a torchvision transform pipeline by name.

    Notes:
        Processed images are already 224×RGB. Normalization is applied only
        in ``train`` / ``eval`` transforms — never baked into saved files.
    """

    if name == "none":
        return None
    if name == "tensor":
        return transforms.ToTensor()
    if name == "eval":
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    if name == "train":
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    raise ValueError(f"Unknown transform selection: {name}")


def create_dataloader(
    split: SplitName | str,
    *,
    config: DatasetConfig | None = None,
    batch_size: int | None = None,
    shuffle: bool | None = None,
    num_workers: int | None = None,
    image_size: int | None = None,
    transform_name: TransformName = "tensor",
    transform: Callable[[Image.Image], Any] | None = None,
) -> DataLoader:
    """Create a configurable DataLoader for a processed split.

    Args:
        split: ``train`` / ``validation`` / ``test``.
        config: Optional dataset configuration.
        batch_size: Override batch size.
        shuffle: Override shuffle (defaults to True for train).
        num_workers: Override worker processes (keep 0 on 8 GB RAM if unsure).
        image_size: Transform resize target.
        transform_name: Built-in transform preset when ``transform`` is None.
        transform: Explicit transform callable (wins over ``transform_name``).
    """

    cfg = config or get_dataset_config()
    split_name = SplitName(split) if not isinstance(split, SplitName) else split

    size = image_size or cfg.image_size
    chosen_transform = transform
    if chosen_transform is None:
        chosen_transform = build_transforms(transform_name, image_size=size)

    dataset: MayaImageDataset = build_split_dataset(
        split_name,
        config=cfg,
        transform=chosen_transform,
    )

    if shuffle is None:
        shuffle = split_name == SplitName.TRAIN

    loader = DataLoader(
        dataset,
        batch_size=batch_size or cfg.batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers if num_workers is None else num_workers,
        pin_memory=cfg.pin_memory,
    )
    logger.info(
        "DataLoader ready split=%s batch=%s shuffle=%s n=%s",
        split_name.value,
        loader.batch_size,
        shuffle,
        len(dataset),
    )
    return loader
