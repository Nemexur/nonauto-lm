from typing import Union, List, Tuple
import torch


class FocalLoss(torch.nn.Module):
    """
    Focal Loss for Seq2Seq model.

    Parameters
    ----------
    alpha : `Union[float, List[float]]`, required
        Alpha factor for Focal Loss.
        - If it's float - binary classification.
        - If it's List[float] - multiclass classification
          and list of floats should match number of classes.
    gamma : `float`, optional (default = `2.0`)
        Gamma factor for Focal Loss.
    size_average : `bool`, optional (default = `True`)
        Whether to average over batch or not.
    """
    def __init__(
        self,
        alpha: Union[float, List[float]],
        gamma: float = 2.0,
        size_average: bool = True
    ) -> None:
        super().__init__()
        self._alpha = alpha
        self._gamma = float(gamma)
        self._size_average = size_average

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        weights: torch.FloatTensor
    ) -> torch.Tensor:
        # weights_batch_sum ~ (batch_size,)
        weights_batch_sum = torch.einsum("b...->b", weights)
        # logits_flat ~ (batch_size * timesteps, num_classes)
        logits_flat = logits.view(-1, logits.size(-1))
        preds = torch.log_softmax(logits_flat, dim=-1)
        # targets_flat ~ (batch_size * timesteps,)
        targets_flat = target.view(-1, 1).long()
        # loss ~ (batch_size * timesteps,)
        nll_loss = -torch.gather(preds, dim=1, index=targets_flat)
        # Add gamma
        if self._gamma is not None:
            weights = self._add_gamma(logits_flat, targets_flat, weights, target.size())
        # Add alpha
        if self._alpha is not None:
            weights = self._add_alpha(targets_flat, weights, target.size())
        # nll_loss ~ (batch_size, timesteps)
        nll_loss = nll_loss.view(*target.size()) * weights
        # per_batch_nll_loss ~ (batch_size,)
        per_batch_nll_loss = (
            torch.einsum("b...->b", nll_loss) / torch.clamp(weights_batch_sum, min=1e-13)
        )
        if self._size_average:
            num_non_empty_sequences = torch.clamp(weights_batch_sum.gt(0).float().sum(), min=1e-13)
            return per_batch_nll_loss.sum() / num_non_empty_sequences
        else:
            return per_batch_nll_loss

    def _add_gamma(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        weights: torch.Tensor,
        gamma_size: Tuple
    ) -> torch.Tensor:
        # probs ~ (batch_size * timesteps, num_classes)
        probs = logits.exp()
        # probs ~ (batch_size * timesteps,)
        probs = torch.gather(probs, dim=1, index=target)
        # gamma_factor ~ (batch_size * timesteps,)
        gamma_factor = (1. - probs) ** self._gamma
        # gamma_factor ~ (batch_size, timesteps)
        gamma_factor = gamma_factor.view(*gamma_size)
        return weights * gamma_factor

    def _add_alpha(
        self,
        target: torch.Tensor,
        weights: torch.Tensor,
        alpha_size: Tuple
    ) -> torch.Tensor:
        # alpha_factor ~ (2,)
        # Only for binary classification
        if isinstance(self._alpha, (float, int)):
            alpha_factor = weights.new_tensor([self._alpha, 1 - self._alpha])
        if isinstance(self._alpha, list):
            # alpha_factor ~ (num_classes,)
            # For multiclass classification
            alpha_factor = weights.new_tensor(self._alpha)
        # alpha_factor ~ (batch_size, timesteps)
        alpha_factor = torch.gather(
            alpha_factor,
            dim=0,
            index=target.view(-1),
        ).view(*alpha_size)
        return weights * alpha_factor


class LabelSmoothingNLL(torch.nn.Module):
    """
    Label Smoothing for NLL Loss.

    Parameters
    ----------
    smoothing : `float`, optional (default = `0.0`)
        Smoothing factor for NLL Loss.
        If smoothing is 0.0 it's a usual NLL Loss.
    size_average : `bool`, optional (default = `True`)
        Whether to size average loss or not.
    """

    def __init__(
        self,
        smoothing: float = 0.0,
        size_average: bool = True
    ) -> None:
        super().__init__()
        self._smoothing = smoothing
        self._size_average = size_average

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        weights: torch.FloatTensor
    ) -> torch.Tensor:
        # logits ~ (batch size, sequence length, num_classes)
        # target ~ (batch size, sequence length)
        # weights ~ (batch size, sequence length)
        # weights_batch_sum ~ (batch_size,)
        weights_batch_sum = torch.einsum("b...->b", weights)
        # logits_flat ~ (batch size * sequence length, num_classes)
        logits_flat = logits.view(-1, logits.size(-1))
        log_probs_flat = torch.log_softmax(logits_flat, dim=-1)
        # targets_flat ~ (batch size * sequence length, 1)
        targets_flat = target.view(-1, 1).long()
        if self._smoothing is not None and self._smoothing > 0.0 and self.training:
            num_classes = logits.size(-1)
            smoothing_value = self._smoothing / num_classes
            # Fill all the correct indices with 1 - smoothing value.
            one_hot_targets = (
                log_probs_flat.new_zeros(log_probs_flat.size())
                .scatter_(-1, targets_flat, 1.0 - self._smoothing)
            )
            smoothed_targets = one_hot_targets + smoothing_value
            nll_loss = -log_probs_flat.mul(smoothed_targets)
            # nll_loss ~ (batch size * sequence length,)
            nll_loss = nll_loss.sum(-1, keepdim=True)
        else:
            # nll_loss ~ (batch size * sequence length,)
            nll_loss = -torch.gather(log_probs_flat, dim=1, index=targets_flat)
        # nll_loss ~ (batch size, sequence length)
        nll_loss = nll_loss.view(*target.size()) * weights
        # per_batch_nll_loss ~ (batch_size,)
        per_batch_nll_loss = (
            torch.einsum("b...->b", nll_loss) / torch.clamp(weights_batch_sum, min=1e-13)
        )
        if self._size_average:
            num_non_empty_sequences = torch.clamp(weights_batch_sum.gt(0).float().sum(), min=1e-13)
            return per_batch_nll_loss.sum() / num_non_empty_sequences
        else:
            return per_batch_nll_loss
