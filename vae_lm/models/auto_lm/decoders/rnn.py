from typing import List, Type, T, Tuple, Dict
import torch
from .decoder import Decoder
from overrides import overrides
from vae_lm.models.base import Embedder
from torch_nlp_utils.data import Vocabulary
from vae_lm.nn.activation import Activation
from vae_lm.nn.feedforward import FeedForward
from vae_lm.nn.recurrent import RecurrentCell


@Decoder.register("auto-recurrent")
class RecurrentDecoder(Decoder):
    def __init__(
        self,
        cell: RecurrentCell,
        embedder: Embedder,
        hidden_sizes: List[int],
        num_classes: int,
        sos_index: int,
        eos_index: int,
        max_timesteps: int,
        teacher_forcing_ratio: float,
        ff_dropout: float = 0.2,
        ff_activation: str = "elu",
        beam_size: int = 2,
        skip_z: bool = False,
    ) -> None:
        super().__init__(
            embedder=embedder,
            sos_index=sos_index,
            eos_index=eos_index,
            max_timesteps=max_timesteps,
            teacher_forcing_ratio=teacher_forcing_ratio,
            beam_size=beam_size,
            skip_z=skip_z,
        )
        self._cell = cell
        self._num_classes = num_classes
        self._projection = FeedForward(
            input_size=(
                self._cell.get_output_size() + self._cell.get_input_size()
                if skip_z else self._cell.get_output_size()
            ),
            num_layers=len(hidden_sizes),
            hidden_sizes=hidden_sizes,
            activations=Activation.by_name(ff_activation),
            dropout=ff_dropout,
            output_size=num_classes,
        )

    def get_input_size(self) -> int:
        return self._cell.get_input_size()

    def get_output_size(self) -> int:
        return self._num_classes

    @overrides
    def decoder_step(
        self,
        step_input: torch.Tensor,
        decoder_state: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # step_input ~ (batch size)
        # decoder_state ~ dict with 3 keys:
        #   latent ~ (batch size, hidden size)
        #   hidden ~ (batch size, hidden size)
        #   cell ~ (batch size, hidden size)
        # step_input ~ (batch size, hidden size)
        step_input = self._embedder(step_input)
        new_hidden, new_cell = self._cell(
            step_input, (decoder_state["hidden"], decoder_state["cell"])
        )
        decoder_state["hidden"], decoder_state["cell"] = new_hidden, new_cell
        logits = self._projection(
            torch.cat((new_hidden, decoder_state["latent"]), dim=-1) if self._skip_z else new_hidden
        )
        return logits, decoder_state

    @classmethod
    def from_params(cls: Type[T], vocab: Vocabulary, **params) -> T:
        cell = RecurrentCell.from_params(**params.pop("cell"))
        embedder = Embedder.from_params(vocab=vocab, **params.pop("embedder"))
        return cls(
            cell=cell,
            embedder=embedder,
            num_classes=vocab.get_vocab_size("target"),
            sos_index=vocab.token_to_index("<sos>", namespace="target"),
            eos_index=vocab.token_to_index("<eos>", namespace="target"),
            **params,
        )
