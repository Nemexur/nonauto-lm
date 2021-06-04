from typing import Type, T, List
from pathlib import Path
import sentencepiece as spm
from torch_nlp_utils.common import FromParams, Params


class SentencePiece(FromParams):
    """Wrapper over SentencePiece model."""
    def __init__(self, model_path: Path, vocab_size: int) -> None:
        super().__init__()
        self._model = spm.SentencePieceProcessor(model_file=str(model_path))
        self._vocab_size = vocab_size

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def encode(self, string: str) -> List[str]:
        """Encode string with SPM Model and return splitted tokens."""
        return self._model.encode(string, out_type=str)

    def id_to_piece(self, idx: int) -> str:
        """Decode index to token."""
        return self._model.id_to_piece(idx)

    @classmethod
    def from_params(cls: Type[T], config: Params) -> T:
        config = Params.from_file(config)
        return cls(
            model_path=f"{config['model_prefix']}.model",
            vocab_size=config["vocab_size"],
        )
