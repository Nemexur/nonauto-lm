from typing import List, Dict, Union, Type, T, NamedTuple, Iterable, Tuple
import torch
from pathlib import Path
from einops import repeat
from collections import OrderedDict
from torch_nlp_utils.data import Vocabulary
# Modules
from .encoders import EncoderOutput
from .losses import LabelSmoothingNLL
from .torch_module import TorchModule
import vae_lm.nn.utils as util
from vae_lm.nn.kl_loss import KLLoss
from vae_lm.nn.kl_scheduler import KLScheduler
from torch_nlp_utils.common import Registrable, Params
from vae_lm.training.metrics import Perplexity, Average


class LatentSample(NamedTuple):
    """Sample from Distribution with mu and sigma."""

    z: torch.Tensor
    mu: torch.Tensor
    sigma: torch.Tensor


class PriorSample(NamedTuple):
    """Named Tuple for Prior Sample."""

    latent: torch.Tensor
    log_prob: torch.Tensor
    mask: torch.Tensor


class PosteriorSample(NamedTuple):
    """Named Tuple for Posterior Sample."""

    latent: LatentSample
    log_prob: torch.Tensor


class VAELmModel(TorchModule, Registrable):
    """
    Generic class for Non-Autoregressive and Autoregressive Variational Language Model.

    Parameters
    ----------
    vocab : `Vocabulary`, required
        There are two typical use-cases for the `Vocabulary` in a `Model`: getting vocabulary sizes
        when constructing embedding matrices or output classifiers (as the vocabulary holds the
        number of classes in your output, also), and translating model output into human-readable
        form.
    kl_scheduler : `KLScheduler`, required
        Scheduler for KL part.
    kl_loss : `KLLoss`, required
        KLLoss module to compute KL-Divergence part.
    iwae : `bool`, optional (default = `False`)
        Whether to use importance weightening or not.
        To use it efficiently number of samples must be greater than 1.
    label_smoothing : `float`, optional (default = `0.0`)
        Label smoothing for NLL Loss. If 0 ordinary NLL Loss is used.

    Inputs
    ------
    src_tokens : `torch.Tensor`, required
        Source tokens for model.
    tgt_tokens : `torch.Tensor`, required
        Target tokens for model.
    manual_kl_step: `bool`, optional (default = `False`)
        Whether to step KL Scheduler manually or not.
    """

    def __init__(
        self,
        vocab: Vocabulary,
        kl_scheduler: KLScheduler,
        kl_loss: KLLoss,
        iwae: bool = False,
        recon_weight: float = 1.0,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self._vocab = vocab
        self._iwae = iwae
        self._kl_scheduler = kl_scheduler
        self._kl_loss = kl_loss
        self._recon_weight = recon_weight
        # Loss
        self._loss = LabelSmoothingNLL(label_smoothing, size_average=False)
        # Metrics
        self._perplexity = Perplexity()
        self._avgs = {metric: Average() for metric in ["avg-kl-weight", "avg-kl", "avg-nll"]}

    @property
    def nsamples_posterior(self) -> int:
        raise NotImplementedError()

    @property
    def is_kl_used(self) -> bool:
        return self._kl_scheduler.kl_weight != 0

    def forward(
        self,
        src_tokens: torch.Tensor,
        tgt_tokens: torch.Tensor,
        manual_kl_step: bool = False,
    ) -> Dict[str, torch.Tensor]:
        # srt_tokens ~ (batch size, seq length)
        # tgt_tokens ~ (batch size, seq length)
        # src_encoded ~ (batch size, seq length, hidden size)
        # src_mask ~ (batch size, seq length)
        batch = src_tokens.size(0)
        src_encoded = self.encode(src_tokens)
        # z ~ (batch size * samples, seq length, hidden size) - for NonAuto
        # z ~ (batch size * samples, hidden size) - for Auto
        # posterior_log_prob ~ (batch size * samples)
        latent, posterior_log_prob = self.sample_from_posterior(src_encoded, random=True)
        # tgt_mask ~ (batch size, seq length)
        tgt_mask = util.get_tokens_mask(tgt_tokens)
        # tgt_mask ~ (batch * samples, seq length)
        tgt_mask = repeat(
            tgt_mask, "batch seq -> (batch samples) seq", samples=self.nsamples_posterior
        )
        # tgt_tokens ~ (batch * samples, seq length)
        tgt_tokens = repeat(
            tgt_tokens, "batch seq -> (batch samples) seq", samples=self.nsamples_posterior
        )
        decoded_output = {
            f"decoder_{x}": self._unwrap_samples(tensor, batch, *tensor.size()[1:])
            for x, tensor in self.decode(latent.z, tgt_mask, target=tgt_tokens).items()
        }
        # prior_log_prob ~ (batch size * samples)
        prior_log_prob = self._get_prior_log_prob(latent, tgt_mask)
        # Losses
        # recon_error ~ (batch size, samples)
        recon_error = decoded_output.pop("decoder_loss")
        # posterior_log_prob ~ (batch size, samples)
        # prior_log_prob ~ (batch size, samples)
        kl_loss_output = {
            f"batch-{x}": self._unwrap_samples(tensor, batch)
            for x, tensor in self._kl_loss(
                posterior_log_prob,
                prior_log_prob,
                kl_weight=self._kl_scheduler.kl_weight,
                latent=latent.z,
            ).items()
        }
        # kl_loss ~ (batch size, samples)
        kl_loss = kl_loss_output.pop("batch-loss")
        # batch_loss ~ (batch size, samples)
        batch_loss = self._recon_weight * recon_error + kl_loss
        if self._iwae:
            # weights ~ (batch size, samples)
            weights = batch_loss.softmax(dim=-1)
            # batch_loss ~ (batch size)
            batch_loss = torch.einsum("bs,bs->b", batch_loss, weights)
        # Step KL Scheduler and recompute KL weight for training
        if not manual_kl_step and self.training:
            self.kl_scheduler_step()
        # Construct output dictionary
        output_dict = {
            "loss_info": {
                "kl-weight": self._kl_scheduler.kl_weight,
                "batch-kl": kl_loss.mean(),
                "batch-nll": recon_error.mean(),
                "batch-loss": batch_loss.mean(),
                **{k: v.mean() for k, v in kl_loss_output.items()},
            },
            "source": src_tokens,
            "target": tgt_tokens,
            **decoded_output,
        }
        # Update metrics
        mean_recon_error = recon_error.mean()
        mean_kl_loss = kl_loss.mean()
        self._perplexity(mean_recon_error)
        for item, value in [
            ("avg-kl-weight", self._kl_scheduler.kl_weight),
            ("avg-kl", mean_kl_loss),
            ("avg-nll", mean_recon_error),
        ]:
            self._avgs[item](value)
        return output_dict

    def kl_scheduler_step(self) -> None:
        self._kl_scheduler.step()

    def _get_prior_log_prob(self, z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Get Log Probability of Prior Distribution based on `z` and its `mask`."""
        raise NotImplementedError()

    def encode(self, tokens: torch.Tensor) -> EncoderOutput:
        """
        Encode `tokens`.

        Returns
        -------
        `EncoderOutput`
            Encoded tokens, context and mask for them.
        """
        raise NotImplementedError()

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
        raise NotImplementedError()

    def sample(
        self,
        samples: int,
        lengths: List[int],
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Get samples from prior distribution and decode them.

        Parameters
        ----------
        samples : `int`, required
            Number of samples to gather.
        lengths : `List[int]`, required
            Lengths of each sample.

        Returns
        -------
        `torch.Tensor`
            Tensor of decoded samples sequences.
        """
        prior_sample = self.sample_from_prior(samples, lengths)
        decoded = self.decode(prior_sample.latent, prior_sample.mask)
        return decoded, prior_sample.log_prob

    def sample_from_prior(
        self,
        samples: int,
        lengths: List[int] = None,
    ) -> PriorSample:
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
        # TODO: Add time to get sampling time
        raise NotImplementedError()

    def sample_from_posterior(self, encoded: EncoderOutput, random: bool = True) -> PosteriorSample:
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
            latent : `LatentSample`
                Sampled latent codes.
            log_prob : `torch.Tensor`
                Log probability for sample.
        """
        raise NotImplementedError()

    def _unwrap_samples(self, x: torch.Tensor, batch: int, *sizes) -> torch.Tensor:
        """Unwrap samples to separate dimension."""
        return x.view(batch, self.nsamples_posterior, *sizes)

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {
            "ppl": self._perplexity.get_metric(reset),
            **{name: metric.get_metric(reset) for name, metric in self._avgs.items()},
        }

    def make_output_human_readable(
        self, output_dict: Dict[str, torch.Tensor], namespace: str = "target"
    ) -> List[str]:
        """
        Takes the result of `forward` and makes it human readable.  Most of the time, the only thing
        this method does is convert tokens / predicted labels from tensors to strings that humans
        might actually understand.  Somtimes you'll also do an argmax or something in here, too, but
        that most often happens in `Model.forward`, before you compute your metrics.
        This method `modifies` the input dictionary, and also `returns` the same dictionary.
        """
        texts = []
        if output_dict["preds"].dim() < 3:
            preds = output_dict["preds"].unsqueeze(-1)
        else:
            preds = output_dict["preds"]
        for sample in preds.tolist():
            # TODO: Simplify decode in Torch NLP Utils
            texts.append([" ".join(self._vocab.decode({namespace: x})[namespace]) for x in sample])
        output_dict["texts"] = texts
        return output_dict

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
        raise NotImplementedError()

    def encoder_parameters(self) -> Iterable[torch.nn.Parameter]:
        """Return parameters for encoder: q(z|x)."""
        raise NotImplementedError()

    def decoder_parameters(self) -> Iterable[torch.nn.Parameter]:
        """Return parameters for decoder: p(x|z)."""
        raise NotImplementedError()

    @classmethod
    def load(
        cls: Type[T],
        params: Params,
        weights: Union[Path, OrderedDict] = None,
        device: int = -1,
    ) -> T:
        """
        Instantiates an already-trained model, based on the experiment
        configuration and some optional overrides.

        Parameters
        ----------
        params : `Dict[str, Any]`, required
            The configuration that was used to train the model. It should definitely
            have a `model` section, and should probably have a `trainer` section
            as well.
        weights_file : `Union[str, OrderedDict]`, optional (default = `None`)
            Weights of already-trained model. If it a string we would load it here
            elif loaded OrderedDict, it needes to be on the passed device.
            If None we instantiate model with random params as usual.
        device : `int`, optional (default = `-1`)
            By default we load the model on the CPU, but if you want to load it
            for GPU usage you can specify the id of your GPU here
        """
        # Construct vocab
        if "vocabulary" not in params:
            raise Exception("No vocabulary found in trained model configuration.")
        vocab: Vocabulary = Vocabulary.from_files(params.get("vocabulary"))
        device = util.int_to_device(device)
        # Load weights if needed
        if isinstance(weights, Path) and weights.exists():
            weights = torch.load(weights, map_location=device)
            weights = weights["model"] if not isinstance(weights, OrderedDict) else weights
        # Construct model
        model = cls.from_params(vocab=vocab, **params.get("model")).to(device)
        if weights:
            model.load_state_dict(weights)
        return model
