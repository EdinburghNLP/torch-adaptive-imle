# -*- coding: utf-8 -*-

import functools

import torch
from torch import Tensor

from imle.noise import BaseNoiseDistribution

from typing import Optional, Callable

import logging

logger = logging.getLogger(__name__)


def sfe(function: Optional[Callable[[Tensor], Tensor]] = None,
        noise_distribution: Optional[BaseNoiseDistribution] = None,
        noise_temperature: float = 1.0):
    if function is None:
        return functools.partial(sfe,
                                 noise_distribution=noise_distribution,
                                 noise_temperature=noise_temperature)

    @functools.wraps(function)
    def wrapper(theta: Tensor, *args):
        class WrappedFunc(torch.autograd.Function):

            @staticmethod
            def forward(ctx, theta: Tensor, *args):
                # [BATCH_SIZE, ...]
                theta_shape = theta.shape

                # ε ∼ ρ(ε)
                if noise_distribution is None:
                    noise = torch.zeros(size=theta_shape, device=theta.device)
                else:
                    noise = noise_distribution.sample(shape=torch.Size(theta_shape)).to(theta.device)

                # [BATCH_SIZE, N_SAMPLES, ...]
                eps = noise * noise_temperature

                perturbed_theta = theta + eps

                # z = f(θ)
                # [BATCH_SIZE, ...]
                z = function(perturbed_theta)
                assert z.shape == theta.shape

                return z

            @staticmethod
            def backward(ctx, dy):
                # Reminder: ∇θ 𝔼[ f(z) ] = 𝔼ₚ₍z;θ₎ [ f(z) ∇θ log p(z;θ) ]

                return dy

        return WrappedFunc.apply(theta, *args)
    return wrapper
