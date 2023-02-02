# -*- coding: utf-8 -*-

import torch

from torch import Tensor
from abc import ABC, abstractmethod

from typing import Optional

import logging

logger = logging.getLogger(__name__)


class BaseTargetDistribution(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def params(self,
               theta: Tensor,
               dy: Optional[Tensor],
               _is_minimization: bool = False) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def process(self,
                theta: Tensor,
                dy: Tensor,
                gradient: Tensor) -> Tensor:
        return gradient


class TargetDistribution(BaseTargetDistribution):
    r"""
    Creates a generator of target distributions parameterized by :attr:`alpha` and :attr:`beta`.

    Example::

        >>> import torch
        >>> target_distribution = TargetDistribution(alpha=1.0, beta=1.0)
        >>> target_distribution.params(theta=torch.tensor([1.0]), dy=torch.tensor([1.0]))
        tensor([2.])

    Args:
        alpha (float): weight of the initial distribution parameters theta
        beta (float): weight of the downstream gradient dy
        do_gradient_scaling (bool): whether to scale the gradient by 1/λ or not
    """
    def __init__(self,
                 alpha: float = 1.0,
                 beta: float = 1.0,
                 do_gradient_scaling: bool = False,
                 eps: float = 1e-7):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.do_gradient_scaling = do_gradient_scaling
        self.eps = eps

    def params(self,
               theta: Tensor,
               dy: Optional[Tensor],
               alpha: Optional[float] = None,
               beta: Optional[float] = None,
               _is_minimization: bool = False) -> Tensor:
        alpha_ = self.alpha if alpha is None else alpha
        beta_ = self.beta if beta is None else beta

        if _is_minimization is True:
            theta_prime = alpha_ * theta + beta_ * (dy if dy is not None else 0.0)
        else:
            theta_prime = alpha_ * theta - beta_ * (dy if dy is not None else 0.0)
        return theta_prime

    def process(self,
                theta: Tensor,
                dy: Tensor,
                gradient_3d: Tensor) -> Tensor:
        scaling_factor = max(self.beta, self.eps)
        res = (gradient_3d / scaling_factor) if self.do_gradient_scaling is True else gradient_3d
        return res


class AdaptiveTargetDistribution(BaseTargetDistribution):
    def __init__(self,
                 initial_alpha: float = 1.0,
                 initial_beta: float = 1.0,
                 initial_grad_norm: float = 1.0,
                 # Pitch: the initial default hyperparams lead to very stable results,
                 # competitive with manually tuned ones -- E.g. try with 1e-3 for this hyperparam
                 beta_update_step: float = 0.0001,
                 beta_update_momentum: float = 0.0,
                 grad_norm_decay_rate: float = 0.9,
                 target_norm: float = 1.0):
        super().__init__()
        self.alpha = initial_alpha
        self.beta = initial_beta

        self.grad_norm = initial_grad_norm
        self.beta_update_step = beta_update_step
        self.beta_update_momentum = beta_update_momentum
        self.previous_beta_update = 0.0
        self.grad_norm_decay_rate = grad_norm_decay_rate
        self.target_norm = target_norm

    def _perturbation_magnitude(self,
                                theta: Tensor,
                                dy: Optional[Tensor]):
        norm_dy = torch.linalg.norm(dy).item() if dy is not None else 1.0
        return 0.0 if norm_dy <= 0.0 else self.beta * (torch.linalg.norm(theta) / norm_dy)

    def params(self,
               theta: Tensor,
               dy: Optional[Tensor],
               _is_minimization: bool = False) -> Tensor:
        pm = self._perturbation_magnitude(theta, dy)
        if _is_minimization is True:
            theta_prime = self.alpha * theta + pm * (dy if dy is not None else 0.0)
        else:
            theta_prime = self.alpha * theta - pm * (dy if dy is not None else 0.0)
        return theta_prime

    def process(self,
                theta: Tensor,
                dy: Tensor,
                gradient_3d: Tensor) -> Tensor:
        batch_size = gradient_3d.shape[0]
        nb_samples = gradient_3d.shape[1]
        pm = self._perturbation_magnitude(theta, dy)

        # We compute an exponentially decaying sum of the gradient norms
        grad_nnz = torch.count_nonzero(gradient_3d).float()
        nb_gradients = batch_size * nb_samples

        # print('GRAD', gradient_3d.shape, 'GRAD NNZ', grad_nnz, batch_size, nb_samples, grad_nnz / nb_gradients)
        # print(gradient_3d[0, 0].int())

        # Running estimate of the gradient norm (number of non-zero elements for every sample)
        self.grad_norm = self.grad_norm_decay_rate * self.grad_norm + \
                         (1.0 - self.grad_norm_decay_rate) * (grad_nnz / nb_gradients)

        # If the gradient norm is lower than 1, we increase beta; otherwise, we decrease beta.
        beta_update_ = (1.0 if self.grad_norm.item() < self.target_norm else - 1.0) * self.beta_update_step
        beta_update = (self.beta_update_momentum * self.previous_beta_update) + beta_update_

        # Enforcing \beta \geq 0
        self.beta = max(self.beta + beta_update, 0.0)
        self.previous_beta_update = beta_update

        # print(f'Gradient norm: {self.grad_norm:.5f}\tBeta: {self.beta:.5f}')

        res = gradient_3d / (pm if pm > 0.0 else 1.0)
        return res
