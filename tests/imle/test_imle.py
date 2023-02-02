#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

import numpy as np

import torch
from torch import nn, Tensor, Size

from imle.imle import imle
from imle.aimle import aimle
from imle.target import TargetDistribution
from imle.noise import BaseNoiseDistribution
from imle.solvers import select_k, mathias_select_k

from typing import Callable, Optional

import pytest

import logging

logger = logging.getLogger(os.path.basename(sys.argv[0]))


class ConstantNoiseDistribution(BaseNoiseDistribution):
    def __init__(self,
                 constant: float = 0.0,
                 device: Optional[torch.device] = None):
        super().__init__()
        self.constant = torch.tensor(constant, dtype=torch.float, device=device)
        self.device = device

    def sample(self,
               shape: Size) -> Tensor:
        return torch.zeros(size=shape, device=self.device) + self.constant


def _test_imle_v1(select_fun: Callable[[Tensor, int], Tensor], nb_samples: int):
    rs = np.random.RandomState(0)

    sym_mismatch_count = 0

    for i in range(2 ** 12):
        input_size = rs.randint(32, 1024)
        k = rs.randint(1, input_size)
        alpha = rs.uniform(0, 10000)
        beta = rs.uniform(0, 10000)

        noise_temperature = 1.0
        target_distribution = TargetDistribution(alpha=alpha, beta=beta)
        noise_distribution = None

        @imle(target_distribution=target_distribution, noise_distribution=noise_distribution, nb_samples=nb_samples,
              theta_noise_temperature=noise_temperature, target_noise_temperature=noise_temperature)
        def imle_select_k(logits: Tensor) -> Tensor:
            return select_fun(logits, k)

        @aimle(target_distribution=target_distribution, noise_distribution=noise_distribution, nb_samples=nb_samples,
               theta_noise_temperature=noise_temperature, target_noise_temperature=noise_temperature,
               symmetric_perturbation=False)
        def aimle_select_k(logits: Tensor) -> Tensor:
            return select_fun(logits, k)

        @aimle(target_distribution=target_distribution, noise_distribution=noise_distribution, nb_samples=nb_samples,
               theta_noise_temperature=noise_temperature, target_noise_temperature=noise_temperature,
               symmetric_perturbation=True)
        def aimle_select_k_sym(logits: Tensor) -> Tensor:
            return select_fun(logits, k)

        init_np = rs.randn(1, input_size)
        linear = nn.Linear(input_size, 1)

        aimle_params = nn.Parameter(torch.tensor(init_np, dtype=torch.float, requires_grad=True), requires_grad=True)
        aimle_sym_params = nn.Parameter(torch.tensor(init_np, dtype=torch.float, requires_grad=True), requires_grad=True)
        imle_params = nn.Parameter(torch.tensor(init_np, dtype=torch.float, requires_grad=True), requires_grad=True)

        aimle_res = linear(aimle_select_k(aimle_params))
        aimle_sym_res = linear(aimle_select_k_sym(aimle_sym_params))
        imle_res = linear(imle_select_k(imle_params))

        res = aimle_res + imle_res + aimle_sym_res

        if nb_samples > 1:
            value = res[0].item()
            for v in res:
                np.testing.assert_allclose(value, v.item(), atol=1e-3, rtol=1e-3)

            res = torch.sum(res)

        res.backward()

        diff = torch.sum(torch.abs(aimle_params.grad - imle_params.grad)).item()
        diff_sym = torch.sum(torch.abs(aimle_sym_params.grad - imle_params.grad)).item()

        assert diff < 1e-24
        sym_mismatch_count += 1 if diff_sym > 1e-24 else 0

    assert sym_mismatch_count > 0


def _test_imle_v2(select_fun: Callable[[Tensor, int], Tensor], nb_samples: int):
    rs = np.random.RandomState(0)

    sym_mismatch_count = 0

    for i in range(2 ** 12):
        input_size = rs.randint(32, 1024)
        k = rs.randint(1, input_size)
        alpha = rs.uniform(0, 10000)
        beta = rs.uniform(0, 10000)

        noise_temperature = rs.uniform(0, 100.0)
        target_distribution = TargetDistribution(alpha=alpha, beta=beta)
        noise_distribution = ConstantNoiseDistribution(constant=rs.uniform(-0.01, -0.01))
        # noise_distribution = ConstantNoiseDistribution(constant=0.0)

        @imle(target_distribution=target_distribution, noise_distribution=noise_distribution, nb_samples=nb_samples,
              theta_noise_temperature=noise_temperature, target_noise_temperature=noise_temperature)
        def imle_select_k(logits: Tensor) -> Tensor:
            return select_fun(logits, k)

        @aimle(target_distribution=target_distribution, noise_distribution=noise_distribution, nb_samples=nb_samples,
               theta_noise_temperature=noise_temperature, target_noise_temperature=noise_temperature,
               symmetric_perturbation=False)
        def aimle_select_k(logits: Tensor) -> Tensor:
            return select_fun(logits, k)

        @aimle(target_distribution=target_distribution, noise_distribution=noise_distribution, nb_samples=nb_samples,
               theta_noise_temperature=noise_temperature, target_noise_temperature=noise_temperature,
               symmetric_perturbation=True)
        def aimle_select_k_sym(logits: Tensor) -> Tensor:
            return select_fun(logits, k)

        init_np = rs.randn(1, input_size)
        linear = nn.Linear(input_size, 1)

        aimle_params = nn.Parameter(torch.tensor(init_np, dtype=torch.float, requires_grad=True), requires_grad=True)
        aimle_sym_params = nn.Parameter(torch.tensor(init_np, dtype=torch.float, requires_grad=True),
                                        requires_grad=True)
        imle_params = nn.Parameter(torch.tensor(init_np, dtype=torch.float, requires_grad=True), requires_grad=True)

        aimle_res = linear(aimle_select_k(aimle_params))
        aimle_sym_res = linear(aimle_select_k_sym(aimle_sym_params))
        imle_res = linear(imle_select_k(imle_params))

        res = aimle_res + imle_res + aimle_sym_res

        if nb_samples > 1:
            value = res[0].item()
            for v in res:
                np.testing.assert_allclose(value, v.item(), atol=1e-3, rtol=1e-3)

            res = torch.sum(res)

        res.backward()

        diff = torch.sum(torch.abs(aimle_params.grad - imle_params.grad)).item()
        diff_sym = torch.sum(torch.abs(aimle_sym_params.grad - imle_params.grad)).item()

        # print(aimle_sym_params.grad)

        assert diff < 1e-24
        sym_mismatch_count += 1 if diff_sym > 1e-24 else 0

    assert sym_mismatch_count > 0


def _test_imle_v3(select_fun: Callable[[Tensor, int], Tensor],
                  nb_samples: int,
                  batch_size: int):
    rs = np.random.RandomState(0)

    sym_mismatch_count = 0

    for i in range(2 ** 12):
        input_size = rs.randint(32, 1024)
        k = rs.randint(1, input_size)
        alpha = rs.uniform(0, 10000)
        beta = rs.uniform(0, 10000)

        noise_temperature = rs.uniform(0, 100.0)
        target_distribution = TargetDistribution(alpha=alpha, beta=beta)
        noise_distribution = ConstantNoiseDistribution(constant=rs.uniform(-0.01, -0.01))
        # noise_distribution = ConstantNoiseDistribution(constant=0.0)

        @imle(target_distribution=target_distribution, noise_distribution=noise_distribution, nb_samples=nb_samples,
              theta_noise_temperature=noise_temperature, target_noise_temperature=noise_temperature)
        def imle_select_k(logits: Tensor) -> Tensor:
            return select_fun(logits, k)

        @aimle(target_distribution=target_distribution, noise_distribution=noise_distribution, nb_samples=nb_samples,
               theta_noise_temperature=noise_temperature, target_noise_temperature=noise_temperature,
               symmetric_perturbation=False)
        def aimle_select_k(logits: Tensor) -> Tensor:
            return select_fun(logits, k)

        @aimle(target_distribution=target_distribution, noise_distribution=noise_distribution, nb_samples=nb_samples,
               theta_noise_temperature=noise_temperature, target_noise_temperature=noise_temperature,
               symmetric_perturbation=True)
        def aimle_select_k_sym(logits: Tensor) -> Tensor:
            return select_fun(logits, k)

        init_np = rs.randn(batch_size, input_size)
        linear = nn.Linear(input_size, 1)

        aimle_params = nn.Parameter(torch.tensor(init_np, dtype=torch.float, requires_grad=True), requires_grad=True)
        aimle_sym_params = nn.Parameter(torch.tensor(init_np, dtype=torch.float, requires_grad=True),
                                        requires_grad=True)
        imle_params = nn.Parameter(torch.tensor(init_np, dtype=torch.float, requires_grad=True), requires_grad=True)

        aimle_res = linear(aimle_select_k(aimle_params))
        aimle_sym_res = linear(aimle_select_k_sym(aimle_sym_params))
        imle_res = linear(imle_select_k(imle_params))

        res = aimle_res + imle_res + aimle_sym_res

        res_2d = res.view(batch_size, nb_samples)

        if nb_samples > 1:
            for i in range(batch_size):
                value = res_2d[i, 0].item()
                for v in res_2d[i, :]:
                    np.testing.assert_allclose(value, v.item(), atol=1e-3, rtol=1e-3)

        res = torch.sum(res)

        res.backward()

        diff = torch.sum(torch.abs(aimle_params.grad - imle_params.grad)).item()
        diff_sym = torch.sum(torch.abs(aimle_sym_params.grad - imle_params.grad)).item()

        # print(aimle_sym_params.grad)

        assert diff < 1e-24
        sym_mismatch_count += 1 if diff_sym > 1e-24 else 0

    assert sym_mismatch_count > 0


def test_imle_v1a():
    _test_imle_v1(select_fun=select_k, nb_samples=1)


def test_imle_v1b():
    _test_imle_v1(select_fun=mathias_select_k, nb_samples=1)


def test_imle_v1c():
    _test_imle_v1(select_fun=mathias_select_k, nb_samples=10)


def test_imle_v2a():
    _test_imle_v2(select_fun=select_k, nb_samples=1)


def test_imle_v2b():
    _test_imle_v2(select_fun=mathias_select_k, nb_samples=1)


def test_imle_v2c():
    _test_imle_v2(select_fun=mathias_select_k, nb_samples=10)


def test_imle_v3a():
    _test_imle_v3(select_fun=select_k, nb_samples=5, batch_size=3)


def test_imle_v3b():
    _test_imle_v3(select_fun=mathias_select_k, nb_samples=5, batch_size=3)


def test_imle_v3c():
    _test_imle_v3(select_fun=mathias_select_k, nb_samples=5, batch_size=3)


if __name__ == '__main__':
    pytest.main([__file__])
    # test_imle_v2a()
    # test_imle_v1c()
    # test_imle_v3a()
