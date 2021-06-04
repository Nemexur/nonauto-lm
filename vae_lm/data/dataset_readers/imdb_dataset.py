from typing import List, Iterable, Any, Dict, Type, T
import re
import torch
import random
from itertools import chain
from torch.nn.utils.rnn import pad_sequence
from vae_lm.data.sentencepiece import SentencePiece
from torch_nlp_utils.data import DatasetReader, CollateBatch, Batch


# Deleted all special symbols from each sample and splitted them on sentences
# train_test_split with test_size = 0.2, seed = 13
# then splitted test with test_size = 0.5, seed = 13
# Picked sentences with num tokens greater than 5
@DatasetReader.register("imdb")
class IMDBDatasetReader(DatasetReader):
    """
    Reader for IMDB Reviews dataset for Language Modelling task.

    Parameters
    ----------
    spm_model : `SentencePiece`, optional (default = `None`)
        SentencePiece model to perform BPE Encoding.
    max_length : `int`, optional (default = `None`)
        Max length of sequence. If None length is not clipped.
    truncate_length : `bool`, optional (default = `False`)
        Whether to truncate number of tokens or just skip it.
    max_prediction_per_seq : `int`, optional (default = `13`)
        Maximum number of masks in sequence.
    masked_lm_prob : `float`, optional (default = `0.15`)
        How much tokens to mask in sequence.
    sample_masking: `bool`, optional (default = `False`)
        Whether to perform tokens masking or not.
    whole_word_masking : `bool`, optional (default = `False`)
        Whether to perform masking on whole word after encoding or not.
    """

    def __init__(
        self,
        spm_model: SentencePiece,
        max_length: int = None,
        truncate_length: bool = False,
        max_predictions_per_seq: int = 7,
        masked_lm_prob: float = 0.15,
        sample_masking: bool = False,
        whole_word_masking: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._sos = "<sos>"
        self._eos = "<eos>"
        self._prefix = "▁"
        self._spm_model = spm_model
        self._max_length = max_length
        self._max_predictions_per_seq = max_predictions_per_seq
        self._masked_lm_prob = masked_lm_prob
        self._sample_masking = sample_masking
        self._whole_word_masking = whole_word_masking
        self._truncate_length = truncate_length
        self._rng = random.Random(13)

    def _read(self, file_path: str) -> Iterable[Dict[str, Any]]:
        with open(file_path, "r", encoding="utf-8") as file:
            for tokens in map(lambda x: x.rstrip("\n"), file):
                if self._max_length is not None:
                    if not self._truncate_length and len(tokens.split()) >= self._max_length:
                        continue
                    elif self._truncate_length:
                        tokens = " ".join(tokens.split()[:self._max_length])
                    else:
                        # We come to this statement only if number of tokens is less than max length
                        # and we should obviously keep this sample.
                        pass
                yield self.item_to_instance(tokens)

    def item_to_instance(self, tokens: str) -> Dict[str, List[str]]:
        if self._spm_model is not None:
            encoded_tokens = self._spm_model.encode(tokens)
            masked_tokens = (
                self.sample_masking(encoded_tokens)
                if self._sample_masking else encoded_tokens
            )
        else:
            # Different objects for tokens and target
            masked_tokens, encoded_tokens = tokens.split(), tokens.split()
        return {
            "tokens": [self._sos] + masked_tokens + [self._eos],
            "target": [self._sos] + encoded_tokens + [self._eos],
        }

    def sample_masking(self, tokens: List[str]) -> List[str]:
        num_changes = 0
        candidates = []
        num_to_predict = min(
            self._max_predictions_per_seq, max(1, int(round(len(tokens) * self._masked_lm_prob)))
        )
        for token in tokens:
            if (
                self._whole_word_masking and len(candidates) >= 1
                and not token.startswith(self._prefix)
            ):
                candidates[-1].append(token)
            else:
                candidates.append([token])
        # Sample possible masking for valid words
        idx_candidates = list(enumerate(candidates))
        self._rng.shuffle(idx_candidates)
        for idx, cand in idx_candidates:
            if num_changes >= num_to_predict:
                break
            # Check if its valid candidate
            # Pick only one sample as receipts are pretty small
            if self._are_valid_tokens(cand):
                candidates[idx] = [self._pick_masking(x) for x in cand]
                num_changes += len(cand)
        return list(chain(*candidates))

    def _pick_masking(self, token: str) -> str:
        # Pick [MASK] 80% of the time
        if self._rng.random() < 0.8:
            return "[MASK]"
        else:
            # Return original token 10% of the time
            if self._rng.random() < 0.5:
                return token
            else:
                # Ger random token from Sentencepiece 10% of the time
                return self._spm_model.id_to_piece(
                    self._rng.randint(0, self._spm_model.vocab_size - 1)
                )

    @staticmethod
    def _are_valid_tokens(tokens: List[str]) -> bool:
        # Construct word
        word = "".join(tokens)
        # Consider one word sequences with SentencePiece Prefix
        if len(word) < 3:
            return False
        # Pick only tokens that are words, doesn't contain special symbols and numbers
        return True if re.search("^[a-zA-Z▁]+$", word) else False

    @classmethod
    def from_params(cls: Type[T], **params) -> T:
        spm_model = params.pop("spm_model")
        if spm_model is not None:
            spm_model = SentencePiece.from_params(**spm_model)
        return cls(
            spm_model=spm_model,
            **params
        )


@CollateBatch.register("imdb")
class IMDBCollateBatch(CollateBatch):
    def __init__(self, batch: Batch) -> None:
        self.src_tokens = pad_sequence(
            [torch.Tensor(x) for x in batch.tokens], batch_first=True
        ).long()
        self.tgt_tokens = pad_sequence(
            [torch.Tensor(x) for x in batch.target], batch_first=True
        ).long()
