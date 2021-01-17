from typing import NamedTuple, Dict, Any
import json
import shutil
import tarfile
import tempfile
from pathlib import Path
from loguru import logger
from copy import deepcopy
from .base import NonAutoLmModel
from contextlib import contextmanager


CONFIG_NAME = "config.json"
WEIGHTS_NAME = "weights.th"
METRICS_NAME = "metrics.json"


class Archive(NamedTuple):
    """ An archive comprises a Model and its experimental config with metrics"""

    model: NonAutoLmModel
    config: Dict[str, Any]
    metrics: Dict[str, float]


def archive_model(
    serialization_dir: Path,
    weights: Path,
    archive_path: Path = None,
) -> None:
    """
    Archive the model weights, its training configuration, and its vocabulary to `model.tar.gz`.

    Parameters
    ----------
    serialization_dir : `Path`, required
        The directory where the weights and vocabulary are written out.
    weights : `Path`, required
        Which weights file to include in the archive. The default is `best.th`.
    archive_path : `str`, optional, (default = `None`)
        A full path to serialize the model to. The default is "model.tar.gz" inside the
        serialization_dir. If you pass a directory here, we'll serialize the model
        to "model.tar.gz" inside the directory.
    """
    # Check weights
    weights_file = weights / "model.pt"
    if not weights_file.exists():
        logger.error(
            f"weights file {weights_file} does not exist, unable to archive model."
        )
        return
    metrics_file = weights / METRICS_NAME
    if not metrics_file.exists():
        logger.error(
            f"metrics file {metrics_file} does not exist, unable to archive model."
        )
        return
    # Check config
    config_file = serialization_dir / CONFIG_NAME
    if not config_file.exists():
        logger.error(
            f"config file {config_file} does not exist, unable to archive model."
        )
    # Check archive path
    if archive_path is not None:
        archive_file = archive_path
        if archive_file.is_dir():
            archive_file = archive_file / "model.tar.gz"
    else:
        archive_file = serialization_dir / "model.tar.gz"
    logger.info(f"Archiving data to {archive_file}.")
    with tarfile.open(archive_file, "w:gz") as archive:
        archive.add(config_file, arcname=CONFIG_NAME)
        archive.add(weights_file, arcname=WEIGHTS_NAME)
        archive.add(str(serialization_dir / "vocabulary"), arcname="vocabulary")


def load_archive(
    archive_file: Path,
    cuda_device: int = -1,
) -> Archive:
    """
    Instantiates an Archive from an archived `tar.gz` file.

    Parameters
    ----------
    archive_file : `Path`, required
        The archive file to load the model from.
    cuda_device : `int`, optional (default = `-1`)
        If `cuda_device` is >= 0, the model will be loaded onto the
        corresponding GPU. Otherwise it will be loaded onto the CPU.
    """
    logger.info(f"Loading archive file {archive_file}")
    tempdir = None
    try:
        if archive_file.is_dir():
            serialization_dir = archive_file
        else:
            with extracted_archive(archive_file, cleanup=False) as tempdir:
                serialization_dir = tempdir
        weights_path = serialization_dir / WEIGHTS_NAME
        # Load config
        with (serialization_dir / CONFIG_NAME).open("r", encoding="utf-8") as file:
            config = json.load(file)
        # Load metrics
        with (serialization_dir / METRICS_NAME).open("r", encoding="utf-8") as file:
            metrics = json.load(file)
        # Instantiate model. Use a duplicate of the config, as it will get consumed.
        model_params = deepcopy(config)
        model_params["vocabulary"] = str(serialization_dir / "vocabulary")
        model = NonAutoLmModel.load(
            model_params,
            weights=str(weights_path),
            device=cuda_device,
        )
    finally:
        if tempdir is not None:
            logger.info(f"Removing temporary unarchived model dir at {tempdir}")
            shutil.rmtree(tempdir, ignore_errors=True)
    return Archive(
        model=model,
        config=config,
        metrics=metrics,
    )


@contextmanager
def extracted_archive(resolved_archive_file, cleanup=True):
    tempdir = None
    try:
        tempdir = tempfile.mkdtemp()
        logger.info(f"Extracting archive file {resolved_archive_file} to temp dir {tempdir}")
        with tarfile.open(resolved_archive_file, "r:gz") as archive:
            archive.extractall(tempdir)
        yield tempdir
    finally:
        if tempdir is not None and cleanup:
            logger.info(f"Removing temporary unarchived model dir at {tempdir}")
            shutil.rmtree(tempdir, ignore_errors=True)
