#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# This is an extended version of gradient-cli.py that supports AIMLE
# Remember to replace gradient-cli.py with this one

import os
import sys

import torch
import numpy as np

from torch import Tensor, nn
import torch.nn.functional as F

from imle.ste import ste as ste
from imle.imle import imle as imle
from imle.aimle import aimle as aimle
from imle.target import BaseTargetDistribution, TargetDistribution, AdaptiveTargetDistribution
from imle.noise import BaseNoiseDistribution, SumOfGammaNoiseDistribution, GumbelNoiseDistribution

from aaai23.synth import distributions, utils, sfe2 as sfe

import argparse
from tqdm import tqdm

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from typing import Optional

import logging

logger = logging.getLogger(os.path.basename(sys.argv[0]))

torch.set_printoptions(profile="full", linewidth=512)


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


def imle_gradient_fun(topk: distributions.TopK,
                      theta_t: Tensor,
                      b_t: Tensor,
                      target_distribution: BaseTargetDistribution,
                      noise_distribution: BaseNoiseDistribution,
                      noise_temperature: float,
                      nb_samples: int,
                      nb_marginal_samples: int) -> Tensor:
    @imle(theta_noise_temperature=noise_temperature, target_noise_temperature=noise_temperature,
          target_distribution=target_distribution, noise_distribution=noise_distribution,
          nb_samples=nb_samples, nb_marginal_samples=nb_marginal_samples)
    def imle_topk_batched(thetas: Tensor) -> Tensor:
        # return torch.stack([topk.map(thetas[i]) for i in range(thetas.shape[0])])
        return topk.map_2d(thetas)

    def imle_topk(theta: Tensor) -> Tensor:
        return imle_topk_batched(theta.view(1, -1)).view(-1)
    theta_t_param = nn.Parameter(theta_t, requires_grad=True)
    z = imle_topk(theta_t_param)
    loss = objective(z, b_t)
    loss.backward()
    return theta_t_param.grad


def aimle_gradient_fun(topk: distributions.TopK,
                       theta_t: Tensor,
                       b_t: Tensor,
                       target_distribution: BaseTargetDistribution,
                       noise_distribution: BaseNoiseDistribution,
                       noise_temperature: float,
                       nb_samples: int,
                       nb_marginal_samples: int,
                       is_symmetric: bool,
                       warmup_steps: int = 0) -> Tensor:
    @aimle(theta_noise_temperature=noise_temperature, target_noise_temperature=noise_temperature,
           target_distribution=target_distribution, noise_distribution=noise_distribution,
           nb_samples=nb_samples, nb_marginal_samples=nb_marginal_samples, symmetric_perturbation=is_symmetric)
    def imle_topk_batched(thetas: Tensor) -> Tensor:
        # return torch.stack([topk.map(thetas[i]) for i in range(thetas.shape[0])])
        return topk.map_2d(thetas)

    def imle_topk(theta: Tensor) -> Tensor:
        return imle_topk_batched(theta.view(1, -1)).view(-1)
    theta_t_param = nn.Parameter(theta_t, requires_grad=True)

    for _ in range(warmup_steps):
        z = imle_topk(theta_t_param)
        loss = objective(z, b_t)
        loss.backward()
        theta_t_param.grad = None

    z = imle_topk(theta_t_param)
    loss = objective(z, b_t)
    loss.backward()
    return theta_t_param.grad


def ste_gradient_fun(topk: distributions.TopK,
                     theta_t: Tensor,
                     b_t: Tensor,
                     noise_distribution: BaseNoiseDistribution,
                     noise_temperature: float,
                     nb_samples: int) -> Tensor:
    @ste(noise_temperature=noise_temperature, noise_distribution=noise_distribution, nb_samples=nb_samples)
    def ste_topk_batched(thetas: Tensor) -> Tensor:
        # return torch.stack([topk.map(thetas[i]) for i in range(thetas.shape[0])])
        return topk.map_2d(thetas)

    def ste_topk(theta: Tensor) -> Tensor:
        return ste_topk_batched(theta.view(1, -1)).view(-1)
    theta_t_param = nn.Parameter(theta_t, requires_grad=True)
    z = ste_topk(theta_t_param)
    loss = objective(z, b_t)
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


def gs_gradient_fun(theta_t: Tensor,
                    b_t: Tensor,
                    nb_samples: Optional[int],
                    tau: float = 1.0,
                    hard: bool = True) -> Tensor:
    # [B, N]
    theta_t_batch = theta_t.view(1, -1).repeat(nb_samples, 1)
    theta_t_batch_param = nn.Parameter(theta_t_batch, requires_grad=True)
    z = F.gumbel_softmax(theta_t_batch_param, tau=tau, hard=hard)
    loss = objective(z, b_t)
    loss.backward()
    return theta_t_batch_param.grad.mean(0)


def main(argv):
    parser = argparse.ArgumentParser('Gradient Estimation', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-k', action='store', type=int, default=10)
    parser.add_argument('-n', action='store', type=int, default=20)
    parser.add_argument('-o', action='store', type=str, default=None)

    parser.add_argument('--min-lambda', action='store', type=float, default=0.0)
    parser.add_argument('--max-lambda', action='store', type=float, default=100.0)
    parser.add_argument('--nb-lambdas', action='store', type=int, default=1001)
    parser.add_argument('--lambdas', type=float, nargs='+', default=[])

    parser.add_argument('--imle-samples', type=int, nargs='+', default=[])
    parser.add_argument('--nb-marginal-samples', action='store', type=int, default=1)

    # The following methods are not sensitive to lambdas
    parser.add_argument('--aimle-samples', type=int, nargs='+', default=[])
    parser.add_argument('--ste-samples', type=int, nargs='+', default=[])
    parser.add_argument('--sfe-samples', type=int, nargs='+', default=[])
    parser.add_argument('--gs-samples', type=int, nargs='+', default=[])

    parser.add_argument('--seeds', '-s', action='store', type=int, default=64)
    parser.add_argument('--warmup-steps', action='store', type=int, default=100)

    parser.add_argument('--momentum', action='store', type=float, default=0.0)
    parser.add_argument('--target', action='store', type=float, default=1.0)
    parser.add_argument('--tau', action='store', type=float, default=1.0)

    parser.add_argument('--threshold', action='store', type=float, default=1e-8)

    args = parser.parse_args(argv)

    device = torch.device('cpu')
    if torch.cuda.is_available():
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
        device = torch.device('cuda')

    torch.set_num_threads(16)

    n = args.n
    k = args.k

    ms = args.nb_marginal_samples

    topk = distributions.TopK(n, k, device=device)
    print(f'Possible states: {topk.states.shape}')

    if len(args.lambdas) > 0:
        lambdas = np.array(args.lambdas)
    else:
        lambdas = np.linspace(args.min_lambda, args.max_lambda, num=args.nb_lambdas)
    warmup_steps = args.warmup_steps

    # noise_distribution = SumOfGammaNoiseDistribution(k=k, nb_iterations=10)
    noise_distribution = GumbelNoiseDistribution()

    lmd_mse_dict = {
        'Method': [],
        '$\\lambda$': [],
        'Cosine Similarity': [],
        'Seed': []
    }

    pdist_ = nn.CosineSimilarity(dim=1, eps=1e-6)
    pdist = lambda x, y: pdist_(x.view(1, -1), y.view(1, -1)).view(-1)[0]

    for i in range(args.seeds):
        print(f'Processing seed {i} ..')
        rng = np.random.RandomState(i)
        theta = rng.randn(n)

        b_t = torch.abs(torch.from_numpy(rng.randn(n)).float().to(device))
        theta_t = torch.tensor(theta, dtype=torch.float, requires_grad=False, device=device)

        true_gradient = true_gradient_fun(topk, theta_t, b_t)

        for lmd in lambdas:
            target_distribution = TargetDistribution(alpha=1.0, beta=lmd, do_gradient_scaling=True)

            for s in args.imle_samples:
                imle_gradient = imle_gradient_fun(topk, theta_t, b_t,
                                                  target_distribution=target_distribution,
                                                  noise_distribution=noise_distribution,
                                                  noise_temperature=1.0,
                                                  nb_samples=s, nb_marginal_samples=ms)

                dist = pdist(true_gradient, imle_gradient)

                lmd_mse_dict['Method'] += [f'IMLE (Forward, $S={s}$)']
                lmd_mse_dict['$\\lambda$'] += [lmd.item()]
                lmd_mse_dict['Cosine Similarity'] += [dist.item()]
                lmd_mse_dict['Seed'] += [i]

                if False:
                    imle_gradient = imle_gradient_fun(topk, theta_t, b_t,
                                                      target_distribution=target_distribution,
                                                      noise_distribution=noise_distribution,
                                                      noise_temperature=1.0,
                                                      nb_samples=1, nb_marginal_samples=ms)

                    sparsity = float((torch.abs(imle_gradient) < args.threshold).sum().item()) / float(imle_gradient.shape[0])

                    lmd_mse_dict['Method'] += [f'IMLE ($S={s}$, $\\mu = {ms}$) Sparsity']
                    lmd_mse_dict['$\\lambda$'] += [lmd.item()]
                    lmd_mse_dict['Cosine Similarity'] += [sparsity]
                    lmd_mse_dict['Seed'] += [i]

                imle_sym_gradient = aimle_gradient_fun(topk, theta_t, b_t,
                                                       target_distribution=target_distribution,
                                                       noise_distribution=noise_distribution,
                                                       noise_temperature=1.0,
                                                       nb_samples=s, nb_marginal_samples=ms,
                                                       is_symmetric=True)

                dist = pdist(true_gradient, imle_sym_gradient)

                lmd_mse_dict['Method'] += [f'IMLE (Central, $S={s}$)']
                lmd_mse_dict['$\\lambda$'] += [lmd.item()]
                lmd_mse_dict['Cosine Similarity'] += [dist.item()]
                lmd_mse_dict['Seed'] += [i]

                if False:
                    imle_sym_gradient = aimle_gradient_fun(topk, theta_t, b_t,
                                                           target_distribution=target_distribution,
                                                           noise_distribution=noise_distribution,
                                                           noise_temperature=1.0,
                                                           nb_samples=1,
                                                           nb_marginal_samples=ms,
                                                           is_symmetric=True)

                    sparsity = float((torch.abs(imle_sym_gradient) < args.threshold).sum().item()) / float(imle_gradient.shape[0])

                    lmd_mse_dict['Method'] += [f'IMLE Sym ({s}, $\\mu = {ms}$) Sparsity']
                    lmd_mse_dict['$\\lambda$'] += [lmd.item()]
                    lmd_mse_dict['Cosine Similarity'] += [sparsity]
                    lmd_mse_dict['Seed'] += [i]

        for s in args.aimle_samples:
            adaptive_target_distribution = AdaptiveTargetDistribution(initial_beta=0.0,
                                                                      beta_update_momentum=args.momentum,
                                                                      beta_update_step=1e-3,
                                                                      target_norm=args.target)
            aimle_gradient = aimle_gradient_fun(topk, theta_t, b_t,
                                                target_distribution=adaptive_target_distribution,
                                                noise_distribution=noise_distribution,
                                                noise_temperature=1.0,
                                                nb_samples=s,
                                                nb_marginal_samples=ms,
                                                is_symmetric=False,
                                                warmup_steps=warmup_steps)

            dist = pdist(true_gradient, aimle_gradient)

            for lmd in lambdas:
                lmd_mse_dict['Method'] += [f'AIMLE (Forward, $S={s}$)']
                lmd_mse_dict['$\\lambda$'] += [lmd]
                lmd_mse_dict['Cosine Similarity'] += [dist.item()]
                lmd_mse_dict['Seed'] += [i]

            adaptive_target_distribution = AdaptiveTargetDistribution(initial_beta=0.0,
                                                                      beta_update_momentum=args.momentum,
                                                                      beta_update_step=1e-3,
                                                                      target_norm=args.target)
            aimle_gradient = aimle_gradient_fun(topk, theta_t, b_t,
                                                target_distribution=adaptive_target_distribution,
                                                noise_distribution=noise_distribution,
                                                noise_temperature=1.0,
                                                nb_samples=s,
                                                nb_marginal_samples=ms,
                                                is_symmetric=True,
                                                warmup_steps=warmup_steps)

            dist = pdist(true_gradient, aimle_gradient)

            for lmd in lambdas:
                lmd_mse_dict['Method'] += [f'AIMLE (Central $S={s}$)']
                lmd_mse_dict['$\\lambda$'] += [lmd.item()]
                lmd_mse_dict['Cosine Similarity'] += [dist.item()]
                lmd_mse_dict['Seed'] += [i]

        for s in args.ste_samples:
            ste_gradient = ste_gradient_fun(topk, theta_t, b_t,
                                            noise_distribution=noise_distribution,
                                            noise_temperature=1.0, nb_samples=s)

            dist = pdist(true_gradient, ste_gradient)

            for lmd in lambdas:
                lmd_mse_dict['Method'] += [f'STE ($S={s}$)']
                lmd_mse_dict['$\\lambda$'] += [lmd.item()]
                lmd_mse_dict['Cosine Similarity'] += [dist.item()]
                lmd_mse_dict['Seed'] += [i]

        for s in args.sfe_samples:
            sfe_gradient = sfe_gradient_fun(topk, theta_t, b_t, s)

            dist = pdist(true_gradient, sfe_gradient)

            for lmd in lambdas:
                lmd_mse_dict['Method'] += [f'SFE ($S={s}$)']
                lmd_mse_dict['$\\lambda$'] += [lmd.item()]
                lmd_mse_dict['Cosine Similarity'] += [dist.item()]
                lmd_mse_dict['Seed'] += [i]

        for s in args.gs_samples:
            gs_gradient = gs_gradient_fun(theta_t, b_t,
                                          nb_samples=s,
                                          tau=args.tau,
                                          hard=False)

            dist = pdist(true_gradient, gs_gradient)

            for lmd in lambdas:
                lmd_mse_dict['Method'] += [f'Gumbel-Softmax ($S={s}$, $\\tau = {args.tau}$)']
                lmd_mse_dict['$\\lambda$'] += [lmd.item()]
                lmd_mse_dict['Cosine Similarity'] += [dist.item()]
                lmd_mse_dict['Seed'] += [i]

            gs_gradient = gs_gradient_fun(theta_t, b_t,
                                          nb_samples=s,
                                          tau=args.tau,
                                          hard=True)

            dist = pdist(true_gradient, gs_gradient)

            for lmd in lambdas:
                lmd_mse_dict['Method'] += [f'Gumbel-Softmax (Hard, $S={s}$, $\\tau = {args.tau}$)']
                lmd_mse_dict['$\\lambda$'] += [lmd.item()]
                lmd_mse_dict['Cosine Similarity'] += [dist.item()]
                lmd_mse_dict['Seed'] += [i]

    if args.o is not None:
        df = pd.DataFrame.from_dict(lmd_mse_dict)
        all_methods_set = {m for m in lmd_mse_dict['Method']}

        # filter_method_lst = ['SFE (1)', 'SFE (10)']
        # filter_method_lst = [m for m in all_methods_set if 'IMLE' in m]
        filter_method_lst = []

        df = df[~df['Method'].isin(filter_method_lst)]

        import matplotlib as mpl
        xdim, ydim = 8, 3
        mpl.rcParams['figure.figsize'] = xdim, ydim
        mpl.rcParams['font.size'] = 13

        g = sns.lineplot(x='$\\lambda$',
                         y="Cosine Similarity",
                         hue="Method",
                         data=df)
        g.set(title='Value of $\\lambda$ $\\times$ Similarity to the true gradient')

        plt.grid()
        plt.xlim(lambdas[0], lambdas[-1])
        plt.ylim(0.0, 1.0)

        # handles, labels = g.get_legend_handles_labels()
        # g.legend(handles=handles[1:], labels=labels[1:])

        g.legend_.set_title(None)

        # plt.show()
        plt.savefig(args.o, bbox_inches='tight')

    print('Done!')


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    main(sys.argv[1:])
