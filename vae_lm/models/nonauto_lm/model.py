from typing import List, Tuple, Dict, Type, T, Iterable
import torch
from einops import repeat
from itertools import chain
import vae_lm.nn.utils as util
from overrides import overrides
from vae_lm.nn.kl_loss import KLLoss
from torch_nlp_utils.data import Vocabulary
from vae_lm.nn.weight_scheduler import WeightScheduler
from vae_lm.models.base import (
    VAELmModel, PriorSample, PosteriorSample, Embedder, LatentSample
)
# Modules
from .priors import Prior
from .posteriors import Posterior
from .decoders import Decoder
from vae_lm.models.base.encoders import Encoder, EncoderOutput


@VAELmModel.register("nonauto")
class NonAutoModel(VAELmModel):
    def __init__(
        self,
        vocab: Vocabulary,
        embedder: Embedder,
        encoder: Encoder,
        decoder: Decoder,
        posterior: Posterior,
        prior: Prior,
        kl_loss: KLLoss,
        recon_scheduler: WeightScheduler,
        kl_scheduler: WeightScheduler,
        iwae: bool = False,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__(
            vocab=vocab,
            kl_loss=kl_loss,
            recon_scheduler=recon_scheduler,
            kl_scheduler=kl_scheduler,
            iwae=iwae,
            label_smoothing=label_smoothing,
        )
        self._embedder = embedder
        self._encoder = encoder
        self._decoder = decoder
        self._posterior = posterior
        self._prior = prior
        # Vocab projection
        self._vocab_projection = torch.nn.Linear(
            self._decoder.get_output_size(),
            vocab.get_vocab_size(namespace="target"),
        )

    @property
    def nsamples_posterior(self) -> int:
        return self._posterior.samples

    @overrides
    def encode(self, tokens: torch.Tensor) -> EncoderOutput:
        """
        Encode `tokens`.

        Returns
        -------
        `EncoderOutput`
            Encoded tokens, context and mask for them.
        """
        embedded_tokens = self._embedder(tokens)
        mask = util.get_tokens_mask(tokens)
        return self._encoder(embedded_tokens, mask)

    @overrides
    def decode(
        self, z: torch.Tensor, mask: torch.Tensor, target: torch.Tensor = None
    ) -> Dict[str, torch.Tensor]:
        """
        Decode sequence from z and mask.

        Parameters
        ----------
        z : `torch.Tensor`, required
            Latent codes.
        mask : `torch.Tensor`, required
            Mask for latent codes
        target : `torch.Tensor`, optional (default = `None`)
            Target sequence if passed in function computes loss.

        Returns
        -------
        `Dict[str, torch.Tensor]`
            logits : `torch.Tensor`
                Logits after decoding.
            probs : `torch.Tensor`
                Softmaxed logits.
            preds : `torch.Tensor`
                Predicted tokens.
            loss : `torch.Tensor`, optional
                Reconstruction error if target is passed.
        """
        # output_dict ~ logits, probs, preds
        logits = self._vocab_projection(self._decoder(z, mask))
        output_dict = {"logits": logits, "probs": torch.softmax(logits, dim=-1)}
        output_dict["preds"] = torch.argmax(output_dict["probs"], dim=-1)
        # Get padding mask
        if target is not None:
            weights = util.get_tokens_mask(target).float()
            loss = self._loss(output_dict["logits"], target, weights=weights)
            output_dict["loss"] = loss
        return output_dict

    @overrides
    def sample_from_prior(
        self, samples: int, lengths: List[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample latent codes from prior distirbution.

        Parameters
        ----------
        samples : `int`, required
            Number of samples to gather.
        lengths : `List[int]`, optional (default = `None`)
            Lengths of each sample. Value can not be None if non-autoregressive model is used.

        Returns
        -------
        `PriorSample`
            z : `torch.Tensor`
                Sampled latent codes.
            log_prob : `torch.Tensor`
                Log probability for sample.
            mask : `torch.Tensor`
                Mask for sampled tensor.
        """
        z, mask = self._prior.sample(samples, lengths)
        z, log_prob = self._posterior.backward(z, mask=mask)
        return PriorSample(z, log_prob, mask)

    @overrides
    def sample_from_posterior(
        self,
        encoded: EncoderOutput,
        random: bool = True
    ) -> PosteriorSample:
        """
        Sample latent codes from posterior distribution.

        Parameters
        ----------
        encoded : `EncoderOutput`, required
            Encoded source sequence with its mask and context.
        random : `bool`, optional (default = `True`)
            Whether to add randomness or not.

        Returns
        -------
        `PosteriorSample`
            latent : `torch.Tensor`
                Sampled latent codes.
            log_prob : `torch.Tensor`
                Log probability for sample.
        """
        posterior_sample = self._posterior(encoded.output, encoded.mask, random=random)
        return PosteriorSample(*posterior_sample)

    @overrides
    def _get_prior_log_prob(self, z: LatentSample, mask: torch.Tensor = None) -> torch.Tensor:
        """Get Log Probability of Prior Distribution based on `z` and its `mask`."""
        return self._prior.log_probability(z, mask)

    @overrides
    def calc_mutual_info(self, src_tokens: torch.Tensor, random: bool = True) -> torch.Tensor:
        """
        Approximate the mutual information between:

        `I(x, z) = E_p(x){E_q(z|x)[log q(z|x)]} - E_p(x){E_q(z|x)[log q(z)]}`

        Parameters
        ----------
        src_tokens : `torch.Tensor`, required
            Input source tokens.
        random : `bool`, optional (default = `True`)
            Whether to perform sampling in posterior or not.
        """
        encoded = self.encode(src_tokens)
        latent, posterior_log_prob = self.sample_from_posterior(encoded, random=True)
        encoded_mask = repeat(
            encoded.mask, "batch seq -> (batch samples) seq", samples=self.nsamples_posterior
        )
        return self._posterior.calc_mutual_info(latent.z, posterior_log_prob, mask=encoded_mask)

    @overrides
    def encoder_parameters(self) -> Iterable[torch.nn.Parameter]:
        """Return parameters for encoder: `q(z|x)`."""
        return chain(self._encoder.parameters(), self._posterior.parameters())

    @overrides
    def decoder_parameters(self) -> Iterable[torch.nn.Parameter]:
        """Return parameters for decoder: `p(x|z)`."""
        return self._decoder.parameters()

    @classmethod
    def from_params(cls: Type[T], vocab: Vocabulary, **params) -> T:
        embedder = Embedder.from_params(vocab=vocab, **params.pop("embedder"))
        encoder = Encoder.from_params(**params.pop("encoder"))
        decoder = Decoder.from_params(**params.pop("decoder"))
        posterior = Posterior.from_params(**params.pop("posterior"))
        prior = Prior.from_params(**params.pop("prior"))
        kl_loss = KLLoss.from_params(prior=prior, **params.pop("kl_loss"))
        recon_scheduler = WeightScheduler.from_params(**params.pop("recon_scheduler"))
        kl_scheduler = WeightScheduler.from_params(**params.pop("kl_scheduler"))
        return cls(
            vocab=vocab,
            embedder=embedder,
            encoder=encoder,
            decoder=decoder,
            posterior=posterior,
            prior=prior,
            kl_loss=kl_loss,
            recon_scheduler=recon_scheduler,
            kl_scheduler=kl_scheduler,
            **params
        )
