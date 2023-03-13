# -*- coding: utf-8 -*-

import functools

import torch
from torch import Tensor

from imle.noise import BaseNoiseDistribution
from imle.target import BaseTargetDistribution, TargetDistribution

from typing import Callable, Optional

import logging

logger = logging.getLogger(__name__)


def aimle(function: Optional[Callable[[Tensor], Tensor]] = None,
          target_distribution: Optional[BaseTargetDistribution] = None,
          noise_distribution: Optional[BaseNoiseDistribution] = None,
          nb_samples: int = 1,
          nb_marginal_samples: int = 1,
          theta_noise_temperature: float = 1.0,
          target_noise_temperature: float = 1.0,
          symmetric_perturbation: bool = False,
          _is_minimization: bool = False):
    r"""Turns a black-box combinatorial solver in an Exponential Family distribution via Perturb-and-MAP and I-MLE [1].

    The theta function (solver) needs to return the solution to the problem of finding a MAP state for a constrained
    exponential family distribution -- this is the case for most black-box combinatorial solvers [2]. If this condition
    is violated though, the result would not hold and there is no guarantee on the validity of the obtained gradients.

    This function can be used directly or as a decorator.

    [1] Mathias Niepert, Pasquale Minervini, Luca Franceschi - Implicit MLE: Backpropagating Through Discrete
    Exponential Family Distributions. NeurIPS 2021 (https://arxiv.org/abs/2106.01798)
    [2] Marin Vlastelica, Anselm Paulus, Vít Musil, Georg Martius, Michal Rolínek - Differentiation of Blackbox
    Combinatorial Solvers. ICLR 2020 (https://arxiv.org/abs/1912.02175)

    Example::

        >>> from imle.aimle import aimle
        >>> from imle.target import TargetDistribution
        >>> from imle.noise import SumOfGammaNoiseDistribution
        >>> target_distribution = TargetDistribution(alpha=0.0, beta=10.0)
        >>> noise_distribution = SumOfGammaNoiseDistribution(k=21, nb_iterations=100)
        >>> @aimle(target_distribution=target_distribution, noise_distribution=noise_distribution, nb_samples=100,
        >>>        theta_noise_temperature=theta_noise_temperature, target_noise_temperature=5.0)
        >>> def aimle_solver(weights_batch: Tensor) -> Tensor:
        >>>     return torch_solver(weights_batch)

    Args:
        function (Callable[[Tensor], Tensor]): black-box combinatorial solver
        target_distribution (Optional[BaseTargetDistribution]): factory for target distributions
        noise_distribution (Optional[BaseNoiseDistribution]): noise distribution
        nb_samples (int): number of noise samples
        nb_marginal_samples (int): number of noise samples used to compute the marginals
        theta_noise_temperature (float): noise temperature for the theta distribution
        target_noise_temperature (float): noise temperature for the target distribution
        symmetric_perturbation (bool): whether it uses the symmetric version of IMLE
        _is_minimization (bool): whether MAP is solving an argmin problem
    """
    if target_distribution is None:
        target_distribution = TargetDistribution(alpha=1.0, beta=1.0)

    if function is None:
        return functools.partial(aimle,
                                 target_distribution=target_distribution,
                                 noise_distribution=noise_distribution,
                                 nb_samples=nb_samples,
                                 nb_marginal_samples=nb_marginal_samples,
                                 theta_noise_temperature=theta_noise_temperature,
                                 target_noise_temperature=target_noise_temperature,
                                 symmetric_perturbation=symmetric_perturbation,
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
                # [BATCH_SIZE, N_TOTAL_SAMPLES, ...]
                if noise_distribution is None:
                    noise = torch.zeros(size=torch.Size(perturbed_theta_shape), device=theta.device)
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

                # [BATCH_SIZE * NB_SAMPLES, ...]
                theta_2d = theta.view(batch_size, 1, -1).repeat(1, nb_total_samples, 1).view(dy_shape)
                # θ'_R = θ - λ dy
                target_theta_r_2d = target_distribution.params(theta_2d, dy,
                                                               _is_minimization=_is_minimization)
                # θ'_L = θ + λ dy -- if symmetric_perturbation is False, then this reduces to θ'_L = θ
                target_theta_l_2d = target_distribution.params(theta_2d, - dy if symmetric_perturbation else None,
                                                               _is_minimization=_is_minimization)

                # [BATCH_SIZE, NB_SAMPLES, ...]
                target_theta_r_3d = target_theta_r_2d.view(noise_shape)
                target_theta_l_3d = target_theta_l_2d.view(noise_shape)

                # [BATCH_SIZE, NB_SAMPLES, ...]
                eps = noise * target_noise_temperature

                # [BATCH_SIZE, N_TOTAL_SAMPLES, ...]
                perturbed_target_theta_r_3d = target_theta_r_3d + eps
                perturbed_target_theta_l_3d = target_theta_l_3d + eps

                # [BATCH_SIZE * N_TOTAL_SAMPLES, ...]
                perturbed_target_theta_r_2d = perturbed_target_theta_r_3d.view(dy_shape)
                perturbed_target_theta_l_2d = perturbed_target_theta_l_3d.view(dy_shape)

                # [BATCH_SIZE * N_TOTAL_SAMPLES, ...]

                with torch.inference_mode():
                    # z'_R = MAP(θ'_R + ε)
                    z_r_2d = function(perturbed_target_theta_r_2d)

                    # z'_L = MAP(θ'_L + ε)
                    z_l_2d = function(perturbed_target_theta_l_2d)

                # [BATCH_SIZE, N_TOTAL_SAMPLES, ...]
                z_r_3d = z_r_2d.view(noise_shape)
                z_l_3d = z_l_2d.view(noise_shape)

                if nb_marginal_samples > 1:
                    assert batch_size == z_l_3d.shape[0] == z_r_3d.shape[0]
                    assert nb_total_samples == z_l_3d.shape[1] == z_r_3d.shape[1]

                    # [BATCH_SIZE, N_SAMPLES, N_MARGINAL_SAMPLES, ...]
                    z_l_4d = z_l_3d.view([batch_size, nb_samples, nb_marginal_samples] + list(instance_shape))
                    z_r_4d = z_r_3d.view([batch_size, nb_samples, nb_marginal_samples] + list(instance_shape))

                    z_l_3d = torch.mean(z_l_4d, dim=2)
                    z_r_3d = torch.mean(z_r_4d, dim=2)

                # g = z'_L - z'_R
                # Note that if symmetric_perturbation is False, then z'_L = z
                # [BATCH_SIZE, N_TOTAL_SAMPLES, ...]
                gradient_3d = z_l_3d - z_r_3d

                if symmetric_perturbation is True:
                    gradient_3d = gradient_3d / 2.0

                # [BATCH_SIZE, N_TOTAL_SAMPLES, ...]
                gradient_3d = target_distribution.process(theta, dy, gradient_3d)

                # [BATCH_SIZE, ...]
                gradient = gradient_3d.mean(dim=1)

                return (- gradient) if _is_minimization is True else gradient

        return WrappedFunc.apply(theta, *args)
    return wrapper
