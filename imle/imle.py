# -*- coding: utf-8 -*-

import functools

import torch
from torch import Tensor

from imle.noise import BaseNoiseDistribution
from imle.target import BaseTargetDistribution, TargetDistribution

from typing import Optional, Callable

import logging

logger = logging.getLogger(__name__)


def imle(function: Optional[Callable[[Tensor], Tensor]] = None,
         target_distribution: Optional[BaseTargetDistribution] = None,
         noise_distribution: Optional[BaseNoiseDistribution] = None,
         nb_samples: int = 1,
         nb_marginal_samples: int = 1,
         theta_noise_temperature: float = 1.0,
         target_noise_temperature: float = 1.0,
         _gradient_save_path: Optional[str] = None,
         _is_minimization: bool = False):
    r"""Turns a black-box combinatorial solver in an Exponential Family distribution via Perturb-and-MAP and I-MLE [1].

    The input function (solver) needs to return the solution to the problem of finding a MAP state for a constrained
    exponential family distribution -- this is the case for most black-box combinatorial solvers [2]. If this condition
    is violated though, the result would not hold and there is no guarantee on the validity of the obtained gradients.

    This function can be used directly or as a decorator.

    [1] Mathias Niepert, Pasquale Minervini, Luca Franceschi - Implicit MLE: Backpropagating Through Discrete
    Exponential Family Distributions. NeurIPS 2021 (https://arxiv.org/abs/2106.01798)
    [2] Marin Vlastelica, Anselm Paulus, Vít Musil, Georg Martius, Michal Rolínek - Differentiation of Blackbox
    Combinatorial Solvers. ICLR 2020 (https://arxiv.org/abs/1912.02175)

    Example:

        >>> from imle.imle import imle
        >>> from imle.target import TargetDistribution
        >>> from imle.noise import SumOfGammaNoiseDistribution
        >>> from imle.solvers import select_k
        >>> target_distribution = TargetDistribution(alpha=0.0, beta=10.0)
        >>> noise_distribution = SumOfGammaNoiseDistribution(k=21, nb_iterations=100)
        >>> @imle(target_distribution=target_distribution, noise_distribution=noise_distribution, nb_samples=100,
        >>>       theta_noise_temperature=theta_noise_temperature, target_noise_temperature=5.0)
        >>> def imle_select_k(weights_batch: Tensor) -> Tensor:
        >>>     return select_k(weights_batch, k=10)

    Args:
        function (Callable[[Tensor], Tensor]): black-box combinatorial solver
        target_distribution (Optional[BaseTargetDistribution]): factory for target distributions
        noise_distribution (Optional[BaseNoiseDistribution]): noise distribution
        nb_samples (int): number of noise samples
        nb_marginal_samples (int): number of noise samples used to compute the marginals
        theta_noise_temperature (float): noise temperature for the input distribution
        target_noise_temperature (float): noise temperature for the target distribution
        _gradient_save_path (Optional[str]): save the gradient in a numpy tensor at this path
        _is_minimization (bool): whether MAP is solving an argmin problem
    """
    if target_distribution is None:
        target_distribution = TargetDistribution(alpha=1.0, beta=1.0)

    if function is None:
        return functools.partial(imle,
                                 target_distribution=target_distribution,
                                 noise_distribution=noise_distribution,
                                 nb_samples=nb_samples,
                                 nb_marginal_samples=nb_marginal_samples,
                                 theta_noise_temperature=theta_noise_temperature,
                                 target_noise_temperature=target_noise_temperature,
                                 _gradient_save_path=_gradient_save_path,
                                 _is_minimization=_is_minimization)

    @functools.wraps(function)
    def wrapper(theta: Tensor, *args):
        class WrappedFunc(torch.autograd.Function):

            @staticmethod
            def forward(ctx, theta: Tensor, *args):
                # [BATCH_SIZE, ...]
                theta_shape = theta.shape

                batch_size = theta_shape[0]
                instance_shape = theta_shape[1:]

                nb_total_samples = nb_samples * nb_marginal_samples

                # [BATCH_SIZE, N_TOTAL_SAMPLES, ...]
                perturbed_theta_shape = [batch_size, nb_total_samples] + list(instance_shape)

                # ε ∼ ρ(ε)
                if noise_distribution is None:
                    noise = torch.zeros(size=perturbed_theta_shape, device=theta.device)
                else:
                    noise = noise_distribution.sample(shape=torch.Size(perturbed_theta_shape)).to(theta.device)

                # [BATCH_SIZE, N_TOTAL_SAMPLES, ...]
                eps = noise * theta_noise_temperature

                # [BATCH_SIZE, N_TOTAL_SAMPLES, ...]
                perturbed_theta_3d = theta.view(batch_size, 1, -1).repeat(1, nb_total_samples, 1).view(perturbed_theta_shape)
                perturbed_theta_3d = perturbed_theta_3d + eps

                # [BATCH_SIZE * N_TOTAL_SAMPLES, ...]
                perturbed_theta_2d = perturbed_theta_3d.view([-1] + perturbed_theta_shape[2:])

                perturbed_theta_2d_shape = perturbed_theta_2d.shape
                assert perturbed_theta_2d_shape[0] == batch_size * nb_total_samples

                # z = MAP(θ + ε)
                # [BATCH_SIZE * N_TOTAL_SAMPLES, ...]
                z_2d = function(perturbed_theta_2d)
                assert z_2d.shape == perturbed_theta_2d_shape

                # [BATCH_SIZE, N_TOTAL_SAMPLES, ...]
                z_3d = z_2d.view(perturbed_theta_shape)

                ctx.save_for_backward(theta, noise, z_3d)

                # [BATCH_SIZE * N_TOTAL_SAMPLES, ...]
                return z_2d

            @staticmethod
            def backward(ctx, dy):
                # theta: [BATCH_SIZE, ...]
                # noise: [BATCH_SIZE, N_TOTAL_SAMPLES, ...]
                # z_3d: [BATCH_SIZE, N_TOTAL_SAMPLES, ...]
                theta, noise, z_3d = ctx.saved_tensors

                nb_total_samples = nb_samples * nb_marginal_samples
                assert noise.shape[1] == nb_total_samples

                theta_shape = theta.shape
                instance_shape = theta_shape[1:]

                batch_size = theta_shape[0]

                # dy is [BATCH_SIZE * N_TOTAL_SAMPLES, ...]
                dy_shape = dy.shape
                # noise is [BATCH_SIZE, N_TOTAL_SAMPLES, ...]
                noise_shape = noise.shape

                assert noise_shape == z_3d.shape

                # [BATCH_SIZE * N_TOTAL_SAMPLES, ...]
                theta_2d = theta.view(batch_size, 1, -1).repeat(1, nb_total_samples, 1).view(dy_shape)
                # θ' = θ - λ dy

                target_theta_2d = target_distribution.params(theta_2d, dy, _is_minimization=_is_minimization)

                # [BATCH_SIZE, N_TOTAL_SAMPLES, ...]
                target_theta_3d = target_theta_2d.view(noise_shape)

                # [BATCH_SIZE, N_TOTAL_SAMPLES, ...]
                eps = noise * target_noise_temperature

                # [BATCH_SIZE, N_TOTAL_SAMPLES, ...]
                perturbed_target_theta_3d = target_theta_3d + eps

                # [BATCH_SIZE * N_TOTAL_SAMPLES, ...]
                perturbed_target_theta_2d = perturbed_target_theta_3d.view(dy_shape)

                # z' = MAP(θ' + ε)
                # [BATCH_SIZE * N_TOTAL_SAMPLES, ...]
                z_prime_2d = function(perturbed_target_theta_2d)

                # [BATCH_SIZE, N_TOTAL_SAMPLES, ...]
                z_prime_3d = z_prime_2d.view(noise_shape)

                if nb_marginal_samples > 1:
                    assert batch_size == z_3d.shape[0] == z_prime_3d.shape[0]
                    assert nb_total_samples == z_3d.shape[1] == z_prime_3d.shape[1]

                    # [BATCH_SIZE, N_SAMPLES, N_MARGINAL_SAMPLES, ...]
                    z_4d = z_3d.view([batch_size, nb_samples, nb_marginal_samples] + list(instance_shape))
                    z_prime_4d = z_prime_3d.view([batch_size, nb_samples, nb_marginal_samples] + list(instance_shape))

                    z_3d = torch.mean(z_4d, dim=2)
                    z_prime_3d = torch.mean(z_prime_4d, dim=2)

                # g = z - z'
                # [BATCH_SIZE, N_TOTAL_SAMPLES, ...]
                gradient_3d = (z_3d - z_prime_3d)

                # [BATCH_SIZE, N_TOTAL_SAMPLES, ...]
                gradient_3d = target_distribution.process(theta, dy, gradient_3d)

                if _gradient_save_path is not None:
                    import numpy as np
                    with open(_gradient_save_path, 'wb') as f:
                        np.save(f, gradient_3d.detach().cpu().numpy())

                # [BATCH_SIZE, ...]
                gradient = gradient_3d.mean(dim=1)

                return (- gradient) if _is_minimization is True else gradient

        return WrappedFunc.apply(theta, *args)
    return wrapper
