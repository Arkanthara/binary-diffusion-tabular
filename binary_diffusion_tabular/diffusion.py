from abc import ABC, abstractmethod
from typing import Literal, Optional, Callable, Dict, Tuple
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional import accuracy

from binary_diffusion_tabular.model import BaseModel, SimpleTableGenerator


__all__ = [
    "BinaryDiffusion1D",
    "BaseDiffusion",
    "SCHEDULE",
    "DENOISING_TARGET",
    "SAMPLING_STRATEGY",
    "make_beta_schedule",
    "get_mask_torch",
    "flip_values",
]


SCHEDULE = Literal["linear", "quad", "sigmoid"]
DENOISING_TARGET = Literal["mask", "target", "two_way"]
SAMPLING_STRATEGY = Literal["mask", "target", "two_way"]


def make_beta_schedule(
    schedule: SCHEDULE = "linear",
    n_timesteps: int = 1000,
    start: float = 1e-5,
    end: float = 0.5,
) -> torch.Tensor:
    """Make a beta schedule.

    Args:
        schedule: type of schedule to use. Can be "linear", "quad", "sigmoid".
        n_timesteps: number of timesteps to use.
        start: start value. Defaults to 1e-5. Should be generally close to 0
        end: end value. Defaults to 0.5. Should be close to 0.5

    Returns:
        torch.Tensor:beta schedule.
    """

    if schedule == "linear":
        betas = torch.linspace(start, end, n_timesteps)
    elif schedule == "quad":
        betas = torch.linspace(start**0.5, end**0.5, n_timesteps) ** 2
    elif schedule == "sigmoid":
        betas = torch.linspace(-6, 6, n_timesteps)
        betas = torch.sigmoid(betas) * (end - start) + start
    else:
        raise ValueError("Incorrect beta schedule type")
    return betas


def get_mask_torch(betas: torch.Tensor, shape, device="cuda") -> torch.Tensor:
    """Returns masks for a list of betas, each with a given percentage of 1 values

    Args:
        betas: tensor containing percentages of 1 values for each mask
        shape: shape of each mask
        device: the device for the operation, e.g., 'cuda' or 'cpu'

    Returns:
        Tensor of masks, one for each beta value
    """

    # Move tensors to the specified device
    betas = betas.to(device)

    num_masks = betas.shape[0]
    flattened_shape = torch.prod(torch.tensor(shape)).item()

    random_values = torch.rand((num_masks, flattened_shape), device=device)
    masks = (random_values < betas.unsqueeze(-1)).int()
    return masks.reshape(num_masks, *shape)


def flip_values(val):
    """Function that changes 0 to 1 and 1 to 0"""
    return 1 - val


class BaseDiffusion(nn.Module, ABC):

    def __init__(
        self,
        denoise_model: BaseModel,
    ):
        super().__init__()
        self.model = denoise_model
        self.device = next(self.model.parameters()).device

    @classmethod
    @abstractmethod
    def from_config(cls, denoise_model: BaseModel, config: Dict) -> "BaseDiffusion":
        pass

    @property
    @abstractmethod
    def config(self) -> Dict:
        pass

    @abstractmethod
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict, Dict]:
        pass

    @abstractmethod
    def sample(
        self,
        *,
        model_fn: Optional[Callable] = None,
        y: Optional[torch.Tensor] = None,
        n: int,
    ) -> torch.Tensor:
        pass


class BinaryDiffusion1D(BaseDiffusion):
    """Binary Diffusion 1D model."""

    def __init__(
        self,
        denoise_model: SimpleTableGenerator,
        *,
        schedule: SCHEDULE = "linear",
        n_timesteps: int,
        target: DENOISING_TARGET = "mask",
    ):
        """
        Args:
            denoise_model: denoiser model to use.
            schedule: beta schedule to use. Can be "linear", "quad", "sigmoid".
            n_timesteps: number of timesteps to use.
            size: size of the 1d data
            target: what denoiser model predictions to use. Can be "mask", "target", "two_way".
                    two_way: predict both mask and denoiser target
        """

        super().__init__(denoise_model)
        self.size = denoise_model.data_dim

        if target not in ["mask", "target", "two_way"]:
            raise ValueError("Incorrect target type")

        if target == "two_way" and self.model.out_dim != 2 * self.size:
            raise ValueError(
                "Incorrect target size. For `two_way` diffusion output should be 2*size"
            )

        self.target = target

        self.n_timesteps = n_timesteps
        self.schedule = schedule

        self.loss = F.binary_cross_entropy_with_logits
        self.betas = make_beta_schedule(schedule, n_timesteps, start=1 / self.size).to(
            self.device
        )
        self.flip_values = flip_values
        self.pred_postproc = torch.sigmoid

    @classmethod
    def from_config(
        cls, denoise_model: SimpleTableGenerator, config: Dict
    ) -> "BaseDiffusion":
        return cls(
            denoise_model,
            **config,
        )

    @property
    def config(self) -> Dict:
        return {
            "schedule": self.schedule,
            "n_timesteps": self.n_timesteps,
            "target": self.target,
        }

    @property
    def conditional(self) -> bool:
        return self.model.conditional

    @property
    def classifier_free_guidance(self) -> bool:
        return self.model.classifier_free_guidance

    @property
    def n_classes(self) -> int:
        return self.model.n_classes

    def q_sample(
        self, x_0: torch.Tensor, t: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if mask is None:
            shape = x_0.shape
            beta = torch.tensor([self.betas[t]] * shape[0]).to(self.device)
            mask = get_mask_torch(beta, shape[1:]).to(self.device)
        mask = mask.to(bool).to(self.device)
        x_copy = x_0.clone().to(self.device)
        x_copy[mask] = self.flip_values(x_copy[mask])
        return x_copy

    def p_sample(self, x: torch.Tensor, mask_pred: torch.Tensor):
        mask_pred = mask_pred.to(bool)
        x_out = x.clone()
        x_out[mask_pred] = self.flip_values(x_out[mask_pred])
        return x_out

    def _apply_sampling_strategy(
        self,
        x_t: torch.Tensor,
        pred_target: torch.Tensor,
        pred_mask: torch.Tensor,
        t: int,
        strategy: SAMPLING_STRATEGY = "target",
    ) -> torch.Tensor:
        if strategy == "target":
            return pred_target.float()
        elif strategy == "mask":
            return self.p_sample(x_t, pred_mask)
        elif strategy == "half-half":
            return (
                self.p_sample(x_t, pred_mask)
                if t < self.n_timesteps // 2
                else pred_target.float()
            )

    @torch.inference_mode()
    def p_sample_loop(
        self,
        n: int,
        model_fn: Optional[Callable] = None,
        y: Optional[torch.Tensor] = None,
        timesteps: Optional[int] = None,
        threshold: Optional[float] = None,
        schedule: SCHEDULE = "linear",
        strategy: Optional[SAMPLING_STRATEGY] = None,
    ) -> torch.Tensor:
        if self.target == "two_way" and strategy is None:
            strategy = "target"

        if strategy not in ["target", "mask", "half-half"]:
            raise ValueError("Incorrect strategy type")

        if strategy is not None and self.target != "two_way":
            raise ValueError("Strategy can only be used with two_way target")

        if timesteps is None:
            timesteps = list(range(self.n_timesteps))

        if model_fn is None:
            model_fn = self.model
        else:
            model_fn = partial(model_fn, model=self.model)

        if threshold is not None:
            thresholds = torch.tensor([threshold] * self.n_timesteps).to(self.device)
        else:
            thresholds = make_beta_schedule(schedule, self.n_timesteps, start=1 / self.size).to(self.device)

        x_t = torch.randint(0, 2, size=(n, self.size)).float().to(self.device)
        for t in reversed(timesteps):
            ts = torch.tensor([t] * n).to(self.device)

            pred = model_fn(x_t, ts, y=y)

            if self.target == "two_way":
                pred_target, pred_mask = pred.chunk(2, dim=1)

                pred_mask = self.pred_postproc(pred_mask)
                pred_target = self.pred_postproc(pred_target)

                pred_mask = pred_mask > thresholds[t]
                pred_target = pred_target > thresholds[t]

                x_t = self._apply_sampling_strategy(
                    x_t, pred_target, pred_mask, t, strategy
                )
            elif self.target == "target":
                pred = self.pred_postproc(pred)
                pred = pred > thresholds[t]
                x_t = pred.float()
            else:
                pred = self.pred_postproc(pred)
                pred = pred > thresholds[t]
                x_t = self.p_sample(x_t, pred)

            if t != 0:
                beta = torch.tensor([self.betas[t]] * n).to(self.device)
                mask = get_mask_torch(beta, x_t.shape[1:], self.device)
                x_t = self.q_sample(x_t, t, mask)

        return x_t

    @torch.inference_mode()
    def sample(
        self,
        *,
        model_fn: Optional[Callable] = None,
        y: Optional[torch.Tensor] = None,
        n: int,
        timesteps: Optional[int] = None,
        threshold: Optional[float] = None,
        schedule: SCHEDULE = "linear",
        strategy: SAMPLING_STRATEGY = "target",
    ) -> torch.Tensor:
        """Samples data

        Args:
            model_fn: denoising model to use
            y: optional conditioning to use
            n: number of samples to generate
            timesteps: number of timesteps to use during sampling
            threshold: threshold to use for sampling
            strategy: sampling strategy to use. Choices: target, mask, half-half

        Returns:
            torch.Tensor: sampled data
        """

        x = self.p_sample_loop(
            n=n,
            model_fn=model_fn,
            y=y,
            timesteps=timesteps,
            threshold=threshold,
            schedule=schedule,
            strategy=strategy,
        )
        return x

    def forward(
        self, x: torch.Tensor, y: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict, Dict]:
        """Runs binary diffusion model training step

        Model selects random timesteps, adds binary noise to data samples, runs denoiser and computed losses and
        accuracies

        Args:
            x: input data. Shape (BS, data_dim)

            y: optional conditioning. Shape (BS, ...)

        Returns:
            torch.Tensor, Dict, Dict: training loss, losses to log, accuracies to log
        """

        bs = x.shape[0]
        sample_shape = x.shape[1:]

        # select random t step
        t = torch.randint(0, self.n_timesteps, size=(bs,)).to(self.device)

        # sample mask
        beta = self.betas[t].to(self.device)
        mask = get_mask_torch(beta, sample_shape, self.device)

        x_t = self.q_sample(x, t, mask)
        pred = self.model(x_t, t, y=y)

        if self.target == "mask":
            loss = self.loss(pred, mask.float())
            acc = accuracy(self.pred_postproc(pred), mask, task="binary")
            losses = {"loss": loss}
            accs = {"acc": acc}
        elif self.target == "target":
            loss = self.loss(pred, x)
            acc = accuracy(self.pred_postproc(pred), x, task="binary")
            losses = {"loss": loss}
            accs = {"acc": acc}
        else:
            pred_target, pred_mask = pred.chunk(2, dim=1)
            loss_target = self.loss(pred_target, x)
            loss_mask = self.loss(pred_mask, mask.float())
            loss = loss_target + loss_mask
            acc_target = accuracy(self.pred_postproc(pred_target), x, task="binary")
            acc_mask = accuracy(self.pred_postproc(pred_mask), mask, task="binary")

            losses = {"loss_target": loss_target, "loss_mask": loss_mask}
            accs = {"acc_target": acc_target, "acc_mask": acc_mask}

        return loss, losses, accs
