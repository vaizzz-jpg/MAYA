"""Automated verification for MAYA Phase 2 / 2.5 dataset pipeline artefacts."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai.datasets.dataset import MayaImageDataset, build_split_dataset
from ai.datasets.dataloader import build_transforms, create_dataloader
from ai.datasets.dataset_config import (
    FUTURE_DATASETS,
    ClassName,
    DatasetConfig,
    SplitBudget,
    SplitName,
    get_dataset_config,
)
from ai.datasets.preprocessing import resize_preserve_aspect_rgb
from ai.datasets.utils.fs import count_images, is_supported_image


@pytest.fixture(scope="session")
def processed_ready() -> DatasetConfig:
    """Skip suite if processed corpus has not been built yet."""

    config = DatasetConfig(project_root=ROOT)
    train_real = config.class_dir(SplitName.TRAIN, ClassName.REAL)
    if not train_real.exists() or not any(train_real.glob("*.jpg")):
        pytest.skip(
            "Processed dataset missing. Run: python scripts/run_dataset_pipeline.py"
        )
    return config


def test_configuration_loading() -> None:
    config = get_dataset_config()
    assert config.image_size == 224
    assert config.random_seed == 42
    assert config.batch_size > 0
    assert config.num_workers >= 0
    assert SplitName.TRAIN in list(SplitName)
    assert config.dataset_root.name == "dataset"
    assert config.versions_dir.name == "versions"
    assert "faceforensics++" in FUTURE_DATASETS
    assert "celeb-df" in FUTURE_DATASETS


def test_folder_existence(processed_ready: DatasetConfig) -> None:
    for split in SplitName:
        for cls in ClassName:
            folder = processed_ready.class_dir(split, cls)
            assert folder.exists(), f"Missing {folder}"
            assert count_images(folder, processed_ready.supported_extensions) > 0


def test_dataset_loads_and_labels(processed_ready: DatasetConfig) -> None:
    dataset = build_split_dataset(SplitName.TRAIN, config=processed_ready, transform=None)
    assert len(dataset) > 0

    image, label = dataset[0]
    assert isinstance(image, Image.Image)
    assert image.mode == "RGB"
    assert label in (0, 1)

    path, folder_label = dataset.samples[0]
    assert path.parent.name.upper() in {"REAL", "FAKE"}
    assert folder_label == (0 if path.parent.name.upper() == "REAL" else 1)


def test_label_correctness_across_classes(processed_ready: DatasetConfig) -> None:
    dataset = build_split_dataset(SplitName.TEST, config=processed_ready, transform=None)
    seen = {0: False, 1: False}
    for path, label in dataset.samples:
        expected = 0 if path.parent.name.upper() == "REAL" else 1
        assert label == expected
        seen[label] = True
    assert seen[0] and seen[1]


def test_images_readable_random_sample(processed_ready: DatasetConfig) -> None:
    dataset = MayaImageDataset(processed_ready.split_dir(SplitName.VALIDATION))
    assert len(dataset) > 0
    rng = random.Random(0)
    for index in rng.sample(range(len(dataset)), k=min(8, len(dataset))):
        image, label = dataset[index]
        assert image.size[0] > 0 and image.size[1] > 0
        assert label in (0, 1)


def test_transforms_and_dataloader(processed_ready: DatasetConfig) -> None:
    transform = build_transforms("tensor", image_size=processed_ready.image_size)
    dataset = build_split_dataset(
        SplitName.TEST,
        config=processed_ready,
        transform=transform,
    )
    image, label = dataset[0]
    assert tuple(image.shape) == (3, processed_ready.image_size, processed_ready.image_size)
    assert label in (0, 1)

    loader = create_dataloader(
        SplitName.TEST,
        config=processed_ready,
        batch_size=4,
        shuffle=False,
        num_workers=0,
        transform_name="tensor",
    )
    batch_x, batch_y = next(iter(loader))
    assert batch_x.shape[0] <= 4
    assert batch_x.shape[1:] == (3, processed_ready.image_size, processed_ready.image_size)
    assert batch_y.shape[0] == batch_x.shape[0]


def test_resize_helper_keeps_rgb_square() -> None:
    image = Image.new("RGBA", (320, 200), color=(10, 20, 30, 255))
    out = resize_preserve_aspect_rgb(image, 224)
    assert out.mode == "RGB"
    assert out.size == (224, 224)


def test_random_sample_visualization_paths(processed_ready: DatasetConfig) -> None:
    grid = processed_ready.reports_dir / "sample_grid.png"
    dist = processed_ready.reports_dir / "class_distribution.png"
    assert grid.exists(), "sample_grid.png missing — rerun pipeline"
    assert dist.exists(), "class_distribution.png missing — rerun pipeline"


def test_is_supported_image_helper(tmp_path: Path) -> None:
    good = tmp_path / "a.jpg"
    good.write_bytes(b"not-really-an-image-but-extension-ok")
    bad = tmp_path / "notes.txt"
    bad.write_text("x", encoding="utf-8")
    assert is_supported_image(good, (".jpg", ".png"))
    assert not is_supported_image(bad, (".jpg", ".png"))


def test_future_compatibility_registry() -> None:
    spec = FUTURE_DATASETS["celeb-df"]
    assert "REAL" in spec.expected_layout or "FAKE" in spec.expected_layout
    config = DatasetConfig(project_root=ROOT)
    assert "faceforensics++" in config.future_datasets


def test_pipeline_end_to_end_tiny(tmp_path: Path) -> None:
    """Smoke-test the full engineering pipeline on a tiny synthetic corpus."""

    from ai.datasets.pipeline import run_dataset_pipeline

    raw_real = tmp_path / "dataset" / "raw" / "REAL"
    raw_fake = tmp_path / "dataset" / "raw" / "FAKE"
    raw_real.mkdir(parents=True)
    raw_fake.mkdir(parents=True)

    for index in range(12):
        Image.new("RGB", (64, 48), color=(index * 10, 40, 80)).save(
            raw_real / f"real_{index}.jpg"
        )
        Image.new("RGB", (80, 60), color=(80, 40, index * 10)).save(
            raw_fake / f"fake_{index}.jpg"
        )

    # Corrupted / zero-byte must be skipped, not crash
    (raw_real / "broken.jpg").write_bytes(b"")

    config = DatasetConfig(
        project_root=tmp_path,
        random_seed=7,
        image_size=64,
        train_budget=SplitBudget(4, 4),
        validation_budget=SplitBudget(4, 4),
        test_budget=SplitBudget(4, 4),
        sample_grid_per_class=2,
        batch_size=2,
        compute_version_checksum=True,
    )
    result = run_dataset_pipeline(config)
    assert result.validation_passed
    assert result.dataset_report.exists()
    assert result.integrity_report.exists()
    assert result.sample_grid_png.exists()
    assert result.metadata_path is not None
    assert result.metadata_path.exists()

    meta = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert meta["dataset_version"] == config.dataset_version
    assert meta["random_seed"] == 7
    assert meta["image_counts"]["total"] == 24

    dataset = build_split_dataset(SplitName.TRAIN, config=config, transform=None)
    assert len(dataset) == 8
    image, label = dataset[0]
    assert image.size == (64, 64)
    assert label in (0, 1)

    current = config.versions_dir / "CURRENT"
    assert current.exists()
    assert current.read_text(encoding="utf-8").strip() == config.dataset_version
