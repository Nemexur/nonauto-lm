from typing import Dict, Tuple
import torch
import random
from .beam_search import BeamSearch
from abc import ABC, abstractmethod
from vae_lm.models.base import Embedder
from vae_lm.models.base import TorchModule
from torch_nlp_utils.common import Registrable


class Decoder(ABC, TorchModule, Registrable):
    """Decoder to predict translations from latent z."""

    def __init__(
        self,
        embedder: Embedder,
        sos_index: int,
        eos_index: int,
        max_timesteps: int,
        teacher_forcing_ratio: float,
        beam_size: int = 2,
        skip_z: bool = False,
    ) -> None:
        super().__init__()
        if not 0 <= teacher_forcing_ratio <= 1:
            raise ValueError(
                "teacher_forcing_ratio should be in between 0 and 1."
            )
        self._embedder = embedder
        self._sos_index = sos_index
        self._eos_index = eos_index
        self._max_timesteps = max_timesteps
        self._teacher_forcing_ratio = teacher_forcing_ratio
        self._skip_z = skip_z
        self._beam_search = BeamSearch(eos_index, max_timesteps, beam_size=beam_size)

    def forward(
        self,
        z: torch.Tensor,
        target: torch.Tensor = None
    ) -> torch.Tensor:
        # Get Decoder State
        decoder_state = self._init_decoder_state(z)
        # Make initial prediction
        return (
            self._forward_loop(decoder_state, target)
            if target is not None else self._generate(decoder_state)
        )

    def _forward_loop(self, decoder_state: Dict[str, torch.Tensor], target: torch.Tensor) -> torch.Tensor:
        all_logits = []
        all_predictions = []
        prediction = target.new_full((target.size(0), ), fill_value=self._sos_index).long()
        # Make prediction for each timestep
        for timestep in range(target.size(1)):
            input_choice = self._choose_input(prediction, target[:, timestep])
            logits, decoder_state = self.decoder_step(input_choice, decoder_state)
            scores = torch.softmax(logits, dim=-1)
            prediction = torch.argmax(scores, dim=-1)
            all_logits.append(logits.unsqueeze(1))
            all_predictions.append(prediction.unsqueeze(1))
        # logits ~ (batch size, seq length, vocab size)
        # predictions ~ (batch size, seq length)
        return torch.cat(all_logits, dim=1), torch.cat(all_predictions, dim=1)

    def _generate(self, decoder_state: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        # prediction ~ (batch size)
        prediction = decoder_state["hidden"].new_full(
            (decoder_state.batch_size, ), fill_value=self._sos_index
        ).long()
        # log_probabilities ~ (batch_size, beam_size)
        # predictions ~ (batch_size, beam_size, max_steps)
        return self._beam_search.search(prediction, decoder_state, self._beam_step)

    def _beam_step(
        self,
        step_input: torch.Tensor,
        decoder_state: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logits, decoder_state = self.decoder_step(step_input, decoder_state)
        log_prob = torch.log_softmax(logits, dim=-1)
        return log_prob, decoder_state

    def _choose_input(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        input_choice = target
        if self.training:
            use_teacher_forcing = (
                True if random.random() < self._teacher_forcing_ratio else False
            )
            input_choice = target if use_teacher_forcing else prediction
        return input_choice

    def _init_decoder_state(self, latent: torch.Tensor) -> Dict[str, torch.Tensor]:
        # latent ~ (batch size, hidden size)
        return {
            "latent": latent,
            "hidden": latent,
            "cell": latent.new_zeros(*latent.size()),
        }

    @abstractmethod
    def decoder_step(
        self,
        step_input: torch.Tensor,
        decoder_state: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Perform one decoder step.

        Parameters
        ----------
        step_input : `torch.Tensor`, required
            Input in current timestep.
        decoder_state : `Dict[str, torch.Tensor]`, required
            Current decoder state.

        Returns
        -------
        `Tuple[torch.Tensor, Dict[str, torch.Tensor]]`
            Tuple of two elements:
                1. Logits.
                2. Next decoder state.
        """
        pass
