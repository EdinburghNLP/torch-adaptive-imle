#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# This is an extended version of gradient-cli.py that supports AIMLE
# Remember to replace gradient-cli.py with this one

import os
import sys

import torch
import numpy as np

from torch import Tensor, nn

from aaai23.synth import distributions, utils, sfe2 as sfe

import argparse
from tqdm import tqdm

from typing import Optional

import logging

logger = logging.getLogger(os.path.basename(sys.argv[0]))


def objective(z: Tensor, b_t: Tensor) -> Tensor:
    if len(z.shape) > len(b_t.shape):
        z = z.view(-1)

    # print('Z', z.shape, 'b_t', b_t.shape)
    if z.shape[0] > b_t.shape[0]:
        nb_samples = z.shape[0] // b_t.shape[0]
        # broadcast b_t
        z_2d = z.view(nb_samples, b_t.shape[0])
        b_t_2d = b_t.view(1, -1).repeat(nb_samples, 1)
        res_2d = ((z_2d - b_t_2d) ** 2)
        res_1d = res_2d.sum(1)
        ### res = res_1d.mean()
        res = res_1d.sum()
    else:
        res = ((z - b_t) ** 2).sum()
    return res


def true_gradient_fun(topk: distributions.TopK,
                      theta_t: Tensor,
                      b_t: Tensor) -> Tensor:
    objective_bt = lambda z_: objective(z_, b_t)
    # Expected value of the loss
    exact_obective = lambda _theta: utils.expect_obj(topk, _theta, objective_bt)
    theta_t_param = nn.Parameter(theta_t, requires_grad=True)
    loss = exact_obective(theta_t_param)
    loss.backward()
    return theta_t_param.grad


def sfe_gradient_fun(topk: distributions.TopK,
                     theta_t: Tensor,
                     b_t: Tensor,
                     nb_samples: Optional[int]) -> Tensor:
    rs = np.random.RandomState(0)
    objective_bt = lambda z_: objective(z_, b_t)
    sfe_full = sfe.sfe(topk.sample_f(rs), objective_bt, topk.grad_log_p(topk.marginals), nb_samples)
    theta_t_param = nn.Parameter(theta_t, requires_grad=True)
    z = sfe_full(theta_t_param)
    loss = objective(z, b_t)
    loss.backward()
    return theta_t_param.grad


def main(argv):
    parser = argparse.ArgumentParser('Gradient Estimation', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-k', action='store', type=int, default=1)
    parser.add_argument('-n', action='store', type=int, default=20)

    parser.add_argument('--seeds', '-s', action='store', type=int, default=64)

    args = parser.parse_args(argv)

    device = torch.device('cpu')
    if torch.cuda.is_available():
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
        device = torch.device('cuda')

    torch.set_num_threads(16)

    n = args.n
    k = args.k

    topk = distributions.TopK(n, k, device=device)
    print(f'Possible states: {topk.states.shape}')

    lmd_mse_dict = {
        'Method': [],
        '$\\lambda$': [],
        'Cosine Similarity': [],
        'Seed': []
    }

    pdist = nn.CosineSimilarity(dim=1, eps=1e-6)

    for i in tqdm(range(args.seeds), desc='Seed'):
        rng = np.random.RandomState(i)
        theta = rng.randn(n)

        b_t = torch.abs(torch.from_numpy(rng.randn(n)).float().to(device))
        theta_t = torch.tensor(theta, dtype=torch.float, requires_grad=False, device=device)

        true_gradient = true_gradient_fun(topk, theta_t, b_t)

        print(true_gradient)

    print('Done!')


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    main(sys.argv[1:])
