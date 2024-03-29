from typing import List, Tuple, Dict, Type, T, Iterable
import torch
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


@VAELmModel.register("auto")
class AutoModel(VAELmModel):
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

    @property
    def nsamples_posterior(self) -> int:
        return self._posterior.samples

    def set_samples(self, samples: int) -> None:
        self._posterior.samples = samples

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
        # z ~ (batch size, hidden size)
        # target ~ (batch size, seq length)
        # logits ~ (batch size, seq length, hidden size)
        # predictions ~ (batch size, seq length)
        logits, predictions = self._decoder(z, target)
        output_dict = {
            "logits": logits,
            "probs": torch.softmax(logits, dim=-1),
            "preds": predictions,
        }
        if target is not None:
            # Get padding mask
            weights = util.get_tokens_mask(target).float()
            # Get sequence excluding start of sequence token
            # as it is being predicted by the model
            relevant_target = target[:, 1:].contiguous()
            relevant_weights = weights[:, 1:].contiguous()
            output_dict["loss"] = self._loss(
                output_dict["logits"],
                relevant_target,
                weights=relevant_weights,
            )
        return output_dict

    @overrides
    def sample_from_prior(
        self, samples: int, lengths: List[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample latent codes from prior distribution.

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
        z, log_prob = self._prior.sample(samples)
        return PriorSample(z, log_prob, None)

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
        posterior_sample = self._posterior(encoded.ctx, random=random)
        return PosteriorSample(*posterior_sample)

    @overrides
    def _get_prior_log_prob(self, z: LatentSample, mask: torch.Tensor = None) -> torch.Tensor:
        """Get Log Probability of Prior Distribution based on `z` and its `mask`."""
        return self._prior.log_probability(z)

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
        return self._posterior.calc_mutual_info(latent.z, posterior_log_prob)

    @overrides
    def encoder_parameters(self) -> Iterable[torch.nn.Parameter]:
        """Return parameters for encoder: `q(z|x)`."""
        return chain(
            self._encoder.parameters(), self._posterior.parameters(), self._prior.parameters()
        )

    @overrides
    def decoder_parameters(self) -> Iterable[torch.nn.Parameter]:
        """Return parameters for decoder: `p(x|z)`."""
        return self._decoder.parameters()

    @classmethod
    def from_params(cls: Type[T], vocab: Vocabulary, **params) -> T:
        embedder = Embedder.from_params(vocab=vocab, **params.pop("embedder"))
        encoder = Encoder.from_params(**params.pop("encoder"))
        decoder = Decoder.from_params(vocab=vocab, **params.pop("decoder"))
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
