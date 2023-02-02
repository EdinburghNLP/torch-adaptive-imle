#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

import torch
from torch import Tensor

import numpy as np

from nri.utils import maybe_make_logits_symmetric, map_estimator

from imle.aimle import aimle
from imle.ste import ste
from imle.target import TargetDistribution, AdaptiveTargetDistribution
from imle.noise import BaseNoiseDistribution, SumOfGammaNoiseDistribution, GumbelNoiseDistribution

from tqdm import tqdm

import pytest

import logging

logger = logging.getLogger(os.path.basename(sys.argv[0]))


def to_matrix(nb_nodes, flat):
    A_out = np.zeros(shape=(nb_nodes, nb_nodes))
    counter = 0
    flat = flat.view(-1)
    for i in range(A_out.shape[0]):
        for j in range(A_out.shape[1]):
            if i != j:
                A_out[i, j] = flat[counter]
                counter = counter + 1
    return A_out


def _test_nri_v1(nb_iterations: int):
    A = np.array([
        [0, 8, 5, 0, 0],
        [8, 0, 9, 11, 0],
        [5, 9, 0, 15, 10],
        [0, 11, 15, 0, 7],
        [0, 0, 10, 7, 0]
    ])

    A_nodiag = A[~np.eye(A.shape[0], dtype=bool)].reshape(A.shape[0], -1)

    logits = torch.tensor(A_nodiag, requires_grad=False, dtype=torch.float).view(1, -1, 1)
    logits = maybe_make_logits_symmetric(logits, True)

    print(to_matrix(A.shape[0], logits))

    noise_distribution = SumOfGammaNoiseDistribution(k=A.shape[0], nb_iterations=10, device=logits.device)
    target_distribution = TargetDistribution(alpha=1.0, beta=10.0)

    @aimle(target_distribution=target_distribution,
           noise_distribution=noise_distribution,
           nb_samples=nb_iterations,
           theta_noise_temperature=0.0,
           target_noise_temperature=0.0,
           symmetric_perturbation=False)
    def differentiable_map_estimator(logits_: Tensor) -> Tensor:
        return map_estimator(logits_, True)

    res = differentiable_map_estimator(logits)
    res = res.mean(dim=0, keepdim=True)

    res_flat = res.view(-1)

    A_out = to_matrix(A.shape[0], res_flat)

    # It's the example in here: https://www.baeldung.com/java-spanning-trees-kruskal
    gold = np.array(
        [[0, 1, 0, 0, 0],
         [1, 0, 0, 1, 0],
         [0, 0, 0, 1, 1],
         [0, 1, 1, 0, 0],
         [0, 0, 1, 0, 0]])

    assert np.sum(np.abs(A_out - gold)) < 1e-12


def test_nri_v1():
    for nb_iterations in tqdm(range(100)):
        nb_iterations = nb_iterations + 1
        for _ in range(8):
            _test_nri_v1(nb_iterations)


if __name__ == '__main__':
    pytest.main([__file__])
    # test_nri_v1()
    # test_imle_v1c()
    # test_imle_v3a()
