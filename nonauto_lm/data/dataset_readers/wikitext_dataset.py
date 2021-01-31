from typing import List, Iterable, Any, Dict
import re
import torch
import random
from torch.nn.utils.rnn import pad_sequence
from torch_nlp_utils.data import DatasetReader, CollateBatch, Batch


@DatasetReader.register("wiki-text")
class WikiTextDatasetReader(DatasetReader):
    """
    Reader for WikiText dataset for Language Modelling task.

    Parameters
    ----------
    max_length : `int`, optional (default = `None`)
        Max length of sequence. If None length is not clipped.
    remove_sentencepiece_prefix : `bool`, optional (default = `True`)
        Whether to remove sentencepiece prefix for tokens or not.
    max_prediction_per_seq : `int`, optional (default = `13`)
        Maximum number of masks in sequence.
    masked_lm_prob : `float`, optional (default = `0.15`)
        How much tokens to mask in sequence.
    sample_masking: `bool`, optional (default = `False`)
        Whether to perform tokens masking or not.
    """

    def __init__(
        self,
        max_length: int = None,
        remove_sentencepiece_prefix: bool = False,
        max_predictions_per_seq: int = 13,
        masked_lm_prob: float = 0.15,
        sample_masking: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._sos = "<sos>"
        self._eos = "<eos>"
        self._max_length = max_length
        self._remove_sentencepiece_prefix = remove_sentencepiece_prefix
        self._max_predictions_per_seq = max_predictions_per_seq
        self._masked_lm_prob = masked_lm_prob
        self._sample_masking = sample_masking
        self._rng = random.Random(13)

    def _read(self, file_path: str) -> Iterable[Dict[str, Any]]:
        with open(file_path, "r", encoding="utf-8") as file:
            for tokens in map(lambda x: x.rstrip("\n"), file):
                # Change sentencepiece prefix if needed
                if self._remove_sentencepiece_prefix:
                    tokens = re.sub(r"\▁", " ", tokens, flags=re.I)
                    tokens = re.sub(r"\s+", " ", tokens, flags=re.I).strip()
                # Remove special symbols
                tokens = re.sub(r"[^a-zA-Z0-9\<\>\s]", " ", tokens, flags=re.I)
                # Remove extra spaces
                tokens = re.sub(r"\s+", " ", tokens, flags=re.I).strip()
                # Tokens equals target
                # (tokens would come through some noise in posterior)
                splitted_tokens = (
                    tokens.split()[: self._max_length] if self._max_length else tokens.split()
                )
                if self._sample_masking:
                    source_tokens = self.sample_masking(splitted_tokens)
                else:
                    source_tokens = splitted_tokens
                yield {
                    "tokens": [self._sos] + source_tokens + [self._eos],
                    "target": [self._sos] + splitted_tokens + [self._eos],
                }

    def sample_masking(self, tokens: List[str]) -> List[str]:
        """
        Sampling masking for tokens.

        Parameters
        ----------
        tokens : `List[str]`, required
            List of tokens for each to sample masking form.

        Returns
        -------
        `List[str]`
            Masked sequence.
        """
        num_to_predict = min(
            self._max_predictions_per_seq, max(1, int(round(len(tokens) * self._masked_lm_prob)))
        )
        # Construct candidates and output sequence
        num_changes = 0
        output_tokens = tokens[:]
        cand_indices = [
            idx for token, idx in zip(tokens, range(len(tokens)))
            # Don't mask <unk> and numbers
            if token != "<unk>" and re.match(r"\d+", token) is None
        ]
        # Shuffle
        self._rng.shuffle(cand_indices)
        # Sample possible masking for valid words
        for token, cand_idx in zip(tokens, cand_indices):
            if num_changes >= num_to_predict:
                break
            output_tokens[cand_idx] = self._pick_masking(token)
            num_changes += 1
        return output_tokens

    def _pick_masking(self, token: str) -> str:
        # Pick [MASK] 80% of the time
        return "[MASK]" if self._rng.random() < 0.8 else token


@CollateBatch.register("wiki-text")
class WikiTextCollateBatch(CollateBatch):
    def __init__(self, batch: Batch) -> None:
        self.src_tokens = pad_sequence(
            [torch.Tensor(x) for x in batch.tokens], batch_first=True
        ).long()
        self.tgt_tokens = pad_sequence(
            [torch.Tensor(x) for x in batch.target], batch_first=True
        ).long()
