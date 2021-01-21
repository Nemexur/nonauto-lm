from typing import NamedTuple, Dict, Callable
import os
import json
import wandb
import shutil
import tarfile
import tempfile
from pathlib import Path
from loguru import logger
from functools import wraps
import torch.distributed as dist
from contextlib import contextmanager
import nonauto_lm.training.ddp as ddp
from torch_nlp_utils.common import Params
# Modules
from nonauto_lm.models.base import NonAutoLmModel


CONFIG_NAME = "config.json"
WEIGHTS_NAME = "weights.pt"
METRICS_NAME = "metrics.json"


class Archive(NamedTuple):
    """ An archive comprises a Model and its experimental config with metrics"""

    model: NonAutoLmModel
    config: Params
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
    # Check metrics
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
        config = Params.from_file(str(serialization_dir / CONFIG_NAME))
        # Load metrics
        with (serialization_dir / METRICS_NAME).open("r", encoding="utf-8") as file:
            metrics = json.load(file)
        # Instantiate model. Use a duplicate of the config, as it will get consumed.
        model_params = config.duplicate()
        model_params["vocabulary"] = str(serialization_dir / "vocabulary")
        model = NonAutoLmModel.load(
            model_params,
            weights=weights_path,
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


def configure_world(func: Callable) -> Callable:
    """Decorator to configure Distributed Training world and wandb if needed for function."""

    @wraps(func)
    def wrapper(process_rank: int, config: Params, world_size: int = 1, **kwargs) -> None:
        is_master = process_rank == 0
        use_wandb = config.get("use_wandb", False)
        # Setup world for Distributed Training
        if world_size > 1:
            ddp.setup_world(process_rank, world_size, backend=dist.Backend.NCCL)
        # Run wandb in master process
        if is_master and use_wandb:
            wandb.init(
                project=os.getenv("WANDB_PROJECT_NAME"),
                config=config.as_flat_dict(),
                reinit=True,
                tags=config.pop("tags"),
            )
        # Run function
        result = func(process_rank=process_rank, config=config, world_size=world_size, **kwargs)
        if is_master:
            serialization_dir = Path(config["serialization_dir"])
            # Construct archive in distributed training there
            # because wandb hangs in distributed training mode
            # and we need to finish it manually then.
            archive_model(
                serialization_dir=serialization_dir,
                weights=serialization_dir / "best-model",
            )
            if use_wandb:
                # Save archived model to wandb
                wandb.save(str(serialization_dir / "model.tar.gz"))
                wandb.finish()
        return result

    wrapper.original = func
    return wrapper
