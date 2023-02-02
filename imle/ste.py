# -*- coding: utf-8 -*-

import functools

import torch
from torch import Tensor

from imle.noise import BaseNoiseDistribution

from typing import Optional, Callable

import logging

logger = logging.getLogger(__name__)


def ste(function: Optional[Callable[[Tensor], Tensor]] = None,
        noise_distribution: Optional[BaseNoiseDistribution] = None,
        noise_temperature: float = 1.0,
        nb_samples: int = 1):
    r"""Straight-Through Estimator [1]

    [1] Yoshua Bengio, Nicholas Léonard, Aaron C. Courville - Estimating or Propagating Gradients Through
    Stochastic Neurons for Conditional Computation. CoRR abs/1308.3432 (2013)

    Example:

        >>> from imle.ste import ste
        >>> from imle.target import TargetDistribution
        >>> from imle.noise import SumOfGammaNoiseDistribution
        >>> from imle.solvers import select_k
        >>> noise_distribution = SumOfGammaNoiseDistribution(k=21, nb_iterations=100)
        >>> @ste(noise_distribution=noise_distribution, nb_samples=100, noise_temperature=noise_temperature)
        >>> def imle_select_k(weights_batch: Tensor) -> Tensor:
        >>>     return select_k(weights_batch, k=10)

    Args:
        function (Callable[[Tensor], Tensor]): black-box combinatorial solver
        noise_distribution (Optional[BaseNoiseDistribution]): noise distribution
        nb_samples (int): number of noise samples
        noise_temperature (float): noise temperature for the input distribution
    """
    if function is None:
        return functools.partial(ste,
                                 noise_distribution=noise_distribution,
                                 noise_temperature=noise_temperature,
                                 nb_samples=nb_samples)

    @functools.wraps(function)
    def wrapper(theta: Tensor, *args):
        class WrappedFunc(torch.autograd.Function):

            @staticmethod
            def forward(ctx, theta: Tensor, *args):
                # [BATCH_SIZE, ...]
                theta_shape = theta.shape

                batch_size = theta_shape[0]
                instance_shape = theta_shape[1:]

                # [BATCH_SIZE, N_SAMPLES, ...]
                perturbed_theta_shape = [batch_size, nb_samples] + list(instance_shape)

                # ε ∼ ρ(ε)
                if noise_distribution is None:
                    noise = torch.zeros(size=perturbed_theta_shape, device=theta.device)
                else:
                    noise = noise_distribution.sample(shape=torch.Size(perturbed_theta_shape)).to(theta.device)

                # [BATCH_SIZE, N_SAMPLES, ...]
                eps = noise * noise_temperature

                # perturbed_theta = theta + eps

                # [BATCH_SIZE, N_SAMPLES, ...]
                perturbed_theta_3d = theta.view(batch_size, 1, -1).repeat(1, nb_samples, 1).view(perturbed_theta_shape)
                perturbed_theta_3d = perturbed_theta_3d + eps

                # [BATCH_SIZE * N_SAMPLES, ...]
                perturbed_theta_2d = perturbed_theta_3d.view([-1] + perturbed_theta_shape[2:])

                perturbed_theta_2d_shape = perturbed_theta_2d.shape
                assert perturbed_theta_2d_shape[0] == batch_size * nb_samples

                # z = f(θ)
                # [BATCH_SIZE, ...]
                # z = function(perturbed_theta)
                # assert z.shape == theta.shape

                # [BATCH_SIZE * N_SAMPLES, ...]
                z_2d = function(perturbed_theta_2d)
                assert z_2d.shape == perturbed_theta_2d_shape

                ctx.save_for_backward(theta, noise)

                return z_2d

            @staticmethod
            def backward(ctx, dy):
                # res = dy

                # theta: [BATCH_SIZE, ...]
                # noise: [BATCH_SIZE, N_SAMPLES, ...]
                theta, noise = ctx.saved_tensors

                batch_size = theta.shape[0]
                assert batch_size == noise.shape[0]
                nb_samples = noise.shape[1]

                gradient_shape = dy.shape[1:]

                # [BATCH_SIZE, N_SAMPLES, ...]
                dy_3d_shape = [batch_size, nb_samples] + list(gradient_shape)
                dy_3d = dy.view(dy_3d_shape)

                # [BATCH_SIZE, ...]
                res = dy_3d.mean(1)
                return res

        return WrappedFunc.apply(theta, *args)
    return wrapper
