import os
import torch
from loguru import logger
import torch.distributed as dist
from torch_nlp_utils.common import Params
from nonauto_lm.training.utils import configure_world
from torch_nlp_utils.data import DatasetReader, DataIterator, Vocabulary, Namespace, CollateBatch
# Modules
import nonauto_lm.nn.utils as util
from nonauto_lm.nn.optimizer import Optimizer
from nonauto_lm.training.trainer import Trainer
from nonauto_lm.models.base import NonAutoLmModel
from nonauto_lm.nn.lr_scheduler import LRScheduler


@configure_world
def train_worker(process_rank: int, config: Params, world_size: int = 1) -> None:
    is_distributed = world_size > 1
    # Construct Datasets
    # TODO: Move Vocabulary creation before process spawn
    dataset_type = config["dataset_reader"]["type"]
    dataset_reader = DatasetReader.from_params(**config.pop("dataset_reader"))
    train_dataset = dataset_reader.read(config["train_data_path"])
    valid_dataset = dataset_reader.read(config["valid_data_path"])
    # Construct Vocabulary
    vocab_path = os.path.join(config["serialization_dir"], "vocabulary")
    if not os.path.exists(vocab_path):
        logger.debug(
            f"No Vocabulary found at path: {vocab_path}. "
            f"Then we would construct it from datasets."
        )
        vocab = Vocabulary(
            datasets={"train": train_dataset, "valid": valid_dataset},
            namespaces={
                "tokens": Namespace(processing_type="padding_oov"),
                "target": Namespace(processing_type="padding_oov"),
            },
            dependent_namespaces=[["tokens", "target"]],
        )
        # Save only on master
        if process_rank == 0:
            vocab.save(path=os.path.join(config["serialization_dir"], "vocabulary"))
    else:
        logger.debug(f"Found Vocabulary at path: {vocab_path}, loading it.")
        vocab = Vocabulary.from_files(vocab_path)
    train_dataset.encode_with(vocab)
    valid_dataset.encode_with(vocab)
    # Construct Iterators
    logger.debug("Construct DataIterators.")
    train_dataloader = DataIterator(
        train_dataset,
        collate_fn=CollateBatch.by_name(dataset_type),
        drop_last=True,
        shuffle=True,
        **config["data_loader"],
    )
    valid_dataloader = DataIterator(
        valid_dataset,
        collate_fn=CollateBatch.by_name(dataset_type),
        shuffle=True,
        drop_last=True,
        **config["data_loader"],
    )
    # Construct modules
    logger.debug("Instantiating Modules from config.")
    device = util.int_to_device(config["cuda_devices"][process_rank])
    model = NonAutoLmModel.from_params(vocab=vocab, **config.pop("model")).to(device)
    optimizer = Optimizer.from_params(params=model.parameters(), **config.pop("optimizer"))
    scheduler = LRScheduler.from_params(optimizer=optimizer, **config["trainer"].pop("scheduler"))
    # Instantiate Trainer
    logger.debug("Instantiating Trainer.")
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        distributed=device != torch.device("cpu") and world_size != 1,
        cuda_device=device,
        local_rank=process_rank,
        world_size=world_size,
        serialization_dir=config["serialization_dir"],
        use_wandb=config.get("use_wandb", False),
        **config.pop("trainer"),
    )
    # Let all setup get ready for all workers.
    if is_distributed:
        dist.barrier()
    # Run training
    logger.debug("Run Trainer.")
    trainer.train(
        train_dataloader=train_dataloader,
        validation_dataloader=valid_dataloader,
    )
    if config.get("evaluate_on_test", False):
        logger.info("Evaluating on test.")
        test_data_path = config.get("test_data_path")
        if not test_data_path:
            logger.error(
                "You set evaluate_on_test=True but didn't pass test_data_path to evaluate on."
            )
            return
        test_dataset = dataset_reader.read(config["test_data_path"])
        test_dataset.encode_with(vocab)
        test_dataloader = DataIterator(
            test_dataset,
            collate_fn=CollateBatch.by_name(dataset_type),
            shuffle=False,
            **config["data_loader"],
        )
        # Wait for all processes to get ready to start evaluation.
        if is_distributed:
            dist.barrier()
        trainer.evaluate(test_dataloader, desc="Testing")
    logger.success("Finished!!!")
