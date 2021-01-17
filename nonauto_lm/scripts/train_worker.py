from typing import Iterable, Dict, Any, T, List, Union
import os
import re
import torch
import random
from loguru import logger
from torch.distributed import Backend
from torch.nn.utils.rnn import pad_sequence
from torch_nlp_utils.data import (
    DatasetReader, DataIterator, Batch,
    Vocabulary, Namespace
)
# Modules
import nonauto_lm.ddp as ddp
import nonauto_lm.nn.utils as util
from nonauto_lm.trainer import Trainer
from nonauto_lm.base import NonAutoLmModel
from nonauto_lm.nn.optimizer import Optimizer
from nonauto_lm.nn.lr_scheduler import LRScheduler


class WikiTextDatasetReader(DatasetReader):
    """
    Reader for WikiText dataset for Language Modelling task.

    Parameters
    ----------
    max_length : `int`, optional (default = `None`)
        Max length of sequence. If None length is not clipped.
    remove_sentencepiece_prefix : `bool`, optional (default = `True`)
        Whether to remove sentencepiece prefix for tokens or not.
    """

    def __init__(
        self,
        max_length: int = None,
        remove_sentencepiece_prefix: bool = False,
        max_predictions_per_seq: int = 13,
        masked_lm_prob: float = 0.15,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._eos = "<eos>"
        self._max_length = max_length
        self._remove_sentencepiece_prefix = remove_sentencepiece_prefix
        self._max_predictions_per_seq = max_predictions_per_seq
        self._masked_lm_prob = masked_lm_prob
        self._rng = random.Random(13)

    def _read(self, file_path: str) -> Iterable[Dict[str, Any]]:
        with open(file_path, "r", encoding="utf-8") as file:
            for tokens in map(lambda x: x.rstrip("\n"), file):
                # Change sentencepiece prefix if needed
                if self._remove_sentencepiece_prefix:
                    tokens = re.sub(r"\▁", " ", tokens, flags=re.I)
                    tokens = re.sub(r"\s+", " ", tokens, flags=re.I).strip()
                # Tokens equals target
                # (tokens would come through some noise in posterior)
                splitted_tokens = (
                    tokens.split()[: self._max_length] if self._max_length else tokens.split()
                )
                # masked_tokens = self.sample_masking(splitted_tokens)
                # target = splitted_tokens
                # Add eos token at the end
                yield {
                    "tokens": splitted_tokens + [self._eos],
                    "target": splitted_tokens + [self._eos],
                }

    def sample_masking(self, tokens: List[str]) -> str:
        """
        Sampling masking for tokens.

        Parameters
        ----------
        tokens : `List[str]`, required
            List of tokens for each to sample masking form.

        Returns
        -------
        `str`
            Masked sentence.
        """
        num_to_predict = min(
            self._max_predictions_per_seq, max(1, int(round(len(tokens) * self._masked_lm_prob)))
        )
        # Construct candidates and output sequence
        num_changes = 0
        output_tokens = tokens[:]
        cand_indices = [idx for idx in range(len(tokens))]
        # Shuffle
        self._rng.shuffle(cand_indices)
        # Sample possible masking for valid words
        for token, cand_idx in zip(tokens, cand_indices):
            if num_changes >= num_to_predict:
                break
            output_tokens[cand_idx] = self._pick_masking(token)
            num_changes += 1
        return " ".join(output_tokens)

    def _pick_masking(self, token: str) -> str:
        # Pick [MASK] 80% of the time
        return "[MASK]" if self._rng.random() < 0.8 else token


class CollateBatch:
    def __init__(self, batch: Batch) -> None:
        self.src_tokens = pad_sequence(
            [torch.Tensor(x) for x in batch.tokens], batch_first=True
        ).long()
        # Make padding divisible by 2 and 2 for split
        # if self.src_sents.size(-1) % 4 != 0:
        #     self.src_sents = torch.cat(
        #         [self.src_sents, torch.zeros((self.src_sents.size(0), 4 - self.src_sents.size(-1) % 4)).long()],
        #         dim=-1
        #     ).long()
        # if self.src_sents.size(-1) % 2 != 0:
        #     self.src_sents = torch.cat(
        #         (self.src_sents, torch.zeros(self.src_sents.size(0), 1).long()),
        #         dim=-1
        #     ).long()
        self.tgt_tokens = pad_sequence(
            [torch.Tensor(x) for x in batch.target], batch_first=True
        ).long()
        # Make padding divisible by 2 and 2 for split
        # if self.tgt_sents.size(-1) % 4 != 0:
        #     self.tgt_sents = torch.cat(
        #         [self.tgt_sents, torch.zeros((self.tgt_sents.size(0), 4 - self.tgt_sents.size(-1) % 4)).long()],
        #         dim=-1
        #     ).long()
        # if self.tgt_sents.size(-1) % 2 != 0:
        #     self.tgt_sents = torch.cat(
        #         (self.tgt_sents, torch.zeros(self.tgt_sents.size(0), 1).long()),
        #         dim=-1
        #     ).long()
        # self.src_masks = self.src_sents.ne(0).float()
        # self.tgt_masks = self.tgt_sents.ne(0).float()

    def pin_memory(self) -> T:
        self.__dict__ = {prop: tensor.pin_memory() for prop, tensor in self.__dict__.items()}
        return self

    def to_device(
        self, device: Union[str, torch.device], **extra_params
    ) -> Dict[str, torch.Tensor]:
        """Helper function to send batch to device and convert it to dict."""
        return {
            prop: tensor.to(device=device, **extra_params) for prop, tensor in self.__dict__.items()
        }


def train_worker(process_rank: int, config: Dict[str, Any], world_size: int = 1) -> None:
    # TODO: Add wandb init, watch and config there
    if world_size > 1:
        ddp.setup_world(
            process_rank, world_size,
            backend=Backend.NCCL if config["cuda_device"][process_rank] >= 0 else Backend.GLOO
        )
    # Torch debug
    torch.autograd.set_detect_anomaly(True)
    # Construct Datasets
    # TODO: Move Vocabulary creation before process spawn
    dataset_reader = WikiTextDatasetReader(lazy=False, **config.pop("dataset_reader"))
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
        logger.debug(
            f"Found Vocabulary at path: {vocab_path}, loading it."
        )
        vocab = Vocabulary.from_files(vocab_path)
    train_dataset.encode_with(vocab)
    valid_dataset.encode_with(vocab)
    # Construct Iterators
    logger.debug("Construct DataIterators.")
    train_dataloader = DataIterator(
        train_dataset,
        collate_fn=CollateBatch,
        drop_last=True,
        shuffle=True,
        **config["data_loader"],
    )
    valid_dataloader = DataIterator(
        valid_dataset,
        collate_fn=CollateBatch,
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
        **config.pop("trainer"),
    )
    # Run training
    logger.debug("Run Trainer.")
    trainer.train(
        train_dataloader=train_dataloader,
        validation_dataloader=valid_dataloader,
    )
    logger.debug("Finished!!!")
    if config["evaluate_on_test"]:
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
            collate_fn=CollateBatch,
            shuffle=False,
            **config["data_loader"],
        )
        trainer.evaluate(test_dataloader)
