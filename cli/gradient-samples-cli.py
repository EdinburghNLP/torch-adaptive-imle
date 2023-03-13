#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

import torch
import numpy as np

from torch import Tensor, nn

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

from multiprocessing.pool import Pool
from multiprocessing import freeze_support, Manager

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
           nb_samples=nb_samples, nb_marginal_samples=nb_marginal_samples,
           symmetric_perturbation=is_symmetric)
    def imle_topk_batched(thetas: Tensor) -> Tensor:
        # Thetas is [B, N]
        # print('Thetas', thetas.shape)
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


# def main(argv):
argv = sys.argv[1:]
# freeze_support()

parser = argparse.ArgumentParser('Gradient Estimation', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('-k', action='store', type=int, default=2)
parser.add_argument('-n', action='store', type=int, default=5)
parser.add_argument('-o', action='store', type=str, default='gradient-samples.pdf')

parser.add_argument('--min-samples', action='store', type=int, default=1)
parser.add_argument('--max-samples', action='store', type=int, default=100)
parser.add_argument('--samples', type=int, nargs='+', default=[])

parser.add_argument('--imle-lambdas', type=float, nargs='+', default=[])
parser.add_argument('--nb-marginal-samples', type=int, nargs='+', default=[1])

parser.add_argument('--seeds', '-s', action='store', type=int, default=64)
parser.add_argument('--warmup-steps', action='store', type=int, default=100)
parser.add_argument('--processes', '-p', action='store', type=int, default=0)

args = parser.parse_args(argv)

device = torch.device('cpu')
if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    device = torch.device('cuda')

print('Device: ', device)
torch.set_num_threads(32)

n = args.n
# k = 5
k = args.k

topk = distributions.TopK(n, k, device=device)
print(f'Possible states: {topk.states.shape}')

# theta_lst = []

# lambdas = np.linspace(args.min_lambda, args.max_lambda, num=args.nb_lambdas)
samples = args.samples
if len(samples) < 1:
    samples = list(range(args.min_samples, args.max_samples + 1))

pdist_batch = nn.CosineSimilarity(dim=1, eps=1e-6)
pdist = lambda x, y: pdist_batch(x.view(1, -1), y.view(1, -1))[0]


def run_experiment(i: int, lmd_mse_dict, mutex) -> int:
    rng = np.random.RandomState(i)
    theta = rng.randn(n)
    b_t = torch.abs(torch.from_numpy(rng.randn(n)).float().to(device))
    theta_t = torch.tensor(theta, dtype=torch.float, requires_grad=False, device=device)
    # theta_lst += [theta_t]

    true_gradient = true_gradient_fun(topk, theta_t, b_t)

    # noise_distribution = SumOfGammaNoiseDistribution(k=k, nb_iterations=10)
    noise_distribution = GumbelNoiseDistribution()

    for s in samples:
        print(f'Processing {s} ..')

        for ms in args.nb_marginal_samples:

            for lmd in args.imle_lambdas:
                target_distribution = TargetDistribution(alpha=1.0, beta=lmd, do_gradient_scaling=True)
                imle_gradient = imle_gradient_fun(topk, theta_t, b_t,
                                                  target_distribution=target_distribution,
                                                  noise_distribution=noise_distribution,
                                                  noise_temperature=1.0, nb_samples=s, nb_marginal_samples=ms)
                sim = pdist(true_gradient, imle_gradient)

                with mutex:
                    lmd_mse_dict['Method'] += [f'IMLE (Forward, $\\lambda = {lmd}$)']
                    lmd_mse_dict['Samples'] += [s]
                    lmd_mse_dict['Cosine Similarity'] += [sim.item()]
                    lmd_mse_dict['Seed'] += [i]

                imle_sym_gradient = aimle_gradient_fun(topk, theta_t, b_t,
                                                       target_distribution=target_distribution,
                                                       noise_distribution=noise_distribution,
                                                       noise_temperature=1.0, nb_samples=s, nb_marginal_samples=ms,
                                                       is_symmetric=True)
                sim = pdist(true_gradient, imle_sym_gradient)

                with mutex:
                    lmd_mse_dict['Method'] += [f'IMLE (Central, $\\lambda = {lmd}$)']
                    lmd_mse_dict['Samples'] += [s]
                    lmd_mse_dict['Cosine Similarity'] += [sim.item()]
                    lmd_mse_dict['Seed'] += [i]

            adaptive_target_distribution = AdaptiveTargetDistribution(initial_beta=0.0,
                                                                      beta_update_momentum=0.0,
                                                                      beta_update_step=1e-3)
            aimle_gradient = aimle_gradient_fun(topk, theta_t, b_t,
                                                target_distribution=adaptive_target_distribution,
                                                noise_distribution=noise_distribution,
                                                noise_temperature=1.0, nb_samples=s, nb_marginal_samples=ms,
                                                is_symmetric=False, warmup_steps=args.warmup_steps)
            sim = pdist(true_gradient, aimle_gradient)

            with mutex:
                lmd_mse_dict['Method'] += [f'AIMLE (Forward)']
                lmd_mse_dict['Samples'] += [s]
                lmd_mse_dict['Cosine Similarity'] += [sim.item()]
                lmd_mse_dict['Seed'] += [i]

            adaptive_target_distribution = AdaptiveTargetDistribution(initial_beta=0.0,
                                                                      beta_update_momentum=0.0,
                                                                      beta_update_step=1e-3)
            aimle_gradient = aimle_gradient_fun(topk, theta_t, b_t,
                                                target_distribution=adaptive_target_distribution,
                                                noise_distribution=noise_distribution,
                                                noise_temperature=1.0, nb_samples=s, nb_marginal_samples=ms,
                                                is_symmetric=True, warmup_steps=args.warmup_steps)
            sim = pdist(true_gradient, aimle_gradient)

            with mutex:
                lmd_mse_dict['Method'] += [f'AIMLE (Central)']
                lmd_mse_dict['Samples'] += [s]
                lmd_mse_dict['Cosine Similarity'] += [sim.item()]
                lmd_mse_dict['Seed'] += [i]

            if False:
                adaptive_target_distribution = AdaptiveTargetDistribution(initial_beta=0.0,
                                                                          beta_update_momentum=0.9,
                                                                          beta_update_step=1e-3)
                aimle_gradient = aimle_gradient_fun(topk, theta_t, b_t,
                                                    target_distribution=adaptive_target_distribution,
                                                    noise_distribution=noise_distribution,
                                                    noise_temperature=1.0, nb_samples=s, nb_marginal_samples=ms,
                                                    is_symmetric=False, warmup_steps=args.warmup_steps)
                sim = pdist(true_gradient, aimle_gradient)

                with mutex:
                    lmd_mse_dict['Method'] += [f'AIMLE Mom ($\\mu = {ms}$)']
                    lmd_mse_dict['Samples'] += [s]
                    lmd_mse_dict['Cosine Similarity'] += [sim.item()]
                    lmd_mse_dict['Seed'] += [i]

                adaptive_target_distribution = AdaptiveTargetDistribution(initial_beta=0.0,
                                                                          beta_update_momentum=0.9,
                                                                          beta_update_step=1e-3)
                aimle_gradient = aimle_gradient_fun(topk, theta_t, b_t,
                                                    target_distribution=adaptive_target_distribution,
                                                    noise_distribution=noise_distribution,
                                                    noise_temperature=1.0, nb_samples=s, nb_marginal_samples=ms,
                                                    is_symmetric=True, warmup_steps=args.warmup_steps)
                sim = pdist(true_gradient, aimle_gradient)

                with mutex:
                    lmd_mse_dict['Method'] += [f'AIMLE Mom Sym ($\\mu = {ms}$)']
                    lmd_mse_dict['Samples'] += [s]
                    lmd_mse_dict['Cosine Similarity'] += [sim.item()]
                    lmd_mse_dict['Seed'] += [i]

        ste_gradient = ste_gradient_fun(topk, theta_t, b_t,
                                        noise_distribution=noise_distribution,
                                        noise_temperature=1.0, nb_samples=s)
        sim = pdist(true_gradient, ste_gradient)

        with mutex:
            lmd_mse_dict['Method'] += [f'STE']
            lmd_mse_dict['Samples'] += [s]
            lmd_mse_dict['Cosine Similarity'] += [sim.item()]
            lmd_mse_dict['Seed'] += [i]

        sfe_gradient = sfe_gradient_fun(topk, theta_t, b_t, s)
        sim = pdist(true_gradient, sfe_gradient)

        with mutex:
            lmd_mse_dict['Method'] += [f'SFE']
            lmd_mse_dict['Samples'] += [s]
            lmd_mse_dict['Cosine Similarity'] += [sim.item()]
            lmd_mse_dict['Seed'] += [i]

    return 0


def main():
    manager = Manager()
    md = manager.dict()

    md['Method'] = []
    md['Samples'] = []
    md['Cosine Similarity'] = []
    md['Seed'] = []

    mutex = manager.Lock()

    keys = [(i, md, mutex) for i in list(range(args.seeds))]

    if args.processes > 0:
        pool = Pool(processes=args.processes)
        pool.starmap(run_experiment, keys)
    else:
        for entry in keys:
            run_experiment(*entry)

    lmd_mse_dict = dict()
    for k, v in md.items():
        lmd_mse_dict[k] = v

    import math
    samples_field = lmd_mse_dict['Samples']
    lmd_mse_dict['Samples'] = [int(math.log10(s)) for s in samples_field]

    df = pd.DataFrame.from_dict(lmd_mse_dict)
    all_method_lst = sorted({m for m in lmd_mse_dict['Method']})
    filter_method_lst = [m for m in all_method_lst if 'Sym' in m]

    df = df[~df['Method'].isin(filter_method_lst)]

    # sns.set(rc={'figure.figsize': (12, 8)})

    from matplotlib import rcParams
    # figure size in inches
    rcParams['figure.figsize'] = 8, 4
    rcParams['font.size'] = 13

    g = sns.lineplot(x='Samples',
                     y="Cosine Similarity",
                     hue="Method",
                     data=df)
    g.set(title='Number of samples $\\times$ Similarity to the true gradient')

    ticks = [int(math.log10(s)) for s in samples]
    g.set_xticks(ticks)
    g.set_xticklabels([f'$10^{t}$' for t in ticks])

    plt.xlim(ticks[0], ticks[-1])
    plt.ylim(0.0, 1.0)

    g.set_xlabel("Number of Samples")
    g.set_ylabel("Cosine Similarity")

    plt.grid()

    # handles, labels = g.get_legend_handles_labels()
    # g.legend(handles=handles[1:], labels=labels[1:])

    g.legend_.set_title(None)

    # plt.show()
    plt.savefig(args.o, bbox_inches='tight')

    print(lmd_mse_dict)

    print('Done!')


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    freeze_support()
    main()

