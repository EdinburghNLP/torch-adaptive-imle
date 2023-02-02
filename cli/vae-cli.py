#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

import numpy as np

import torch
import torch.nn.functional as F

from torch import nn, optim, Tensor

from imle.imle import imle
from imle.aimle import aimle
from imle.ste import ste
from imle.target import TargetDistribution, AdaptiveTargetDistribution
from imle.noise import BaseNoiseDistribution, SumOfGammaNoiseDistribution, GumbelNoiseDistribution
from imle.solvers import mathias_select_k

from l2x.torch.utils import set_seed
from l2x.torch.dvae.modules import DiscreteVAE

from torch.utils.data import DataLoader

from torchvision import datasets, transforms

import argparse

import socket
import wandb

from typing import Tuple, Callable, Optional

import logging

logger = logging.getLogger(os.path.basename(sys.argv[0]))


class DifferentiableSelectKModel(nn.Module):
    def __init__(self,
                 diff_fun: Callable[[Tensor], Tensor],
                 fun: Callable[[Tensor], Tensor]):
        super().__init__()
        self.diff_fun = diff_fun
        self.fun = fun

    def forward(self, logits: Tensor) -> Tensor:
        return self.diff_fun(logits) if self.training else self.fun(logits)


def gumbel_loss(logits_2d: Tensor,
                rec_2d: Tensor,
                x_2d: Tensor,
                m: int,
                n: int,
                reduction: str = 'mean') -> Tuple[Tensor, int]:
    batch_size = logits_2d.shape[0]
    input_size = x_2d.shape[1]

    if reduction in {'mean'}:
        reduction_fun = lambda x, dim: torch.mean(x, dim=dim)
    elif reduction in {'sum'}:
        reduction_fun = lambda x, dim: torch.sum(x, dim=dim)
    elif reduction in {'none'}:
        reduction_fun = lambda x, dim: x
    else:
        assert False, f'Unknown reduction function: {reduction}'

    # BCE Loss
    bce_loss_fun = torch.nn.BCEWithLogitsLoss(reduction='none')

    # x_2d is [B, H * W], let's make it [B * S, H * W] to match rec_2d
    nb_samples = rec_2d.shape[0] // x_2d.shape[0]
    x_2d_ = x_2d.view(batch_size, 1, input_size)
    x_2d_ = x_2d_.repeat(1, nb_samples, 1)
    x_2d_ = x_2d_.view(batch_size * nb_samples, input_size)

    # Per-pixel BCE Loss -- [B * S, M * N]
    bce_loss = bce_loss_fun(rec_2d, x_2d_)

    # Sum over pixels -- [B * S]
    bce_loss = bce_loss.sum(dim=1)

    # XXX IN THE FOLLOWING I THINK WE SHOULD SUM OVER "S" AND REDUNCTION_FUN OVER "B"
    # This fixes the problem of having to sum over the samples
    if nb_samples > 1:
        bce_loss_2d = bce_loss.view(-1, nb_samples)
        bce_loss = bce_loss_2d.sum(dim=1)

    # Average over batch -- []
    bce_loss = reduction_fun(bce_loss, 0)

    # KL Loss -- [B, M, N]
    logits_3d = logits_2d.view(batch_size, m, n)
    # q_y(b, m), distribution over binary values for N -- [B, M, N]
    q_y = torch.softmax(logits_3d, dim=-1)
    # log q_y: log(q_y(b, m)) -- [B, M, N]
    log_q_y = torch.log(q_y + 1e-20)
    # q_y * (log q_y - log(1/n)) = KL(q_y, 1/n) -- [B, M, N]
    kl_3d = q_y * (log_q_y - np.log(1.0 / n))  # This is equivalent to  q_y * torch.log((q_y + 1e-20) * n)
    # Sum of the KL divergence terms over pixels, i.e. M and N -- [B]
    kl_1d = kl_3d.sum(dim=(1, 2))

    # Average of the KL terms over the batch -- []
    # kl = kl_1d.mean(dim=0)
    kl = reduction_fun(kl_1d, 0)

    loss = bce_loss + kl
    return loss, nb_samples


def main(argv):
    parser = argparse.ArgumentParser('PyTorch I-MLE/DVAE', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--batch-size', '-b', action='store', type=int, default=100, help='Batch Size')
    parser.add_argument('--input-dim', action='store', type=int, default=28 * 28)
    parser.add_argument('--epochs', '-e', action='store', type=int, default=100)

    parser.add_argument('--select-k', '-K', action='store', type=int, default=10, help='Select K')

    parser.add_argument('--code-m', action='store', type=int, default=20)
    parser.add_argument('--code-n', action='store', type=int, default=20)

    parser.add_argument("--method", "-M", type=str, choices=['gumbel-softmax', 'imle', 'aimle', 'ste'], default='imle')

    # Gumbel SoftMax
    parser.add_argument('--anneal-rate', action='store', type=float, default=0.00003)
    parser.add_argument('--init-temperature', action='store', type=float, default=1.0)
    parser.add_argument('--min-temperature', action='store', type=float, default=0.5)
    parser.add_argument('--hard', action='store_true', default=False)

    # AIMLE
    parser.add_argument('--aimle-symmetric', action='store_true', default=False)
    parser.add_argument('--aimle-target', type=str, choices=['standard', 'adaptive'], default='standard')
    parser.add_argument('--aimle-beta-update-step', action='store', type=float, default=0.0001)
    parser.add_argument('--aimle-beta-update-momentum', action='store', type=float, default=0.0)
    parser.add_argument('--aimle-target-norm', action='store', type=float, default=1.0)

    # IMLE
    parser.add_argument('--imle-noise', type=str, choices=['none', 'sog', 'gumbel'], default='sog')
    parser.add_argument('--imle-samples', action='store', type=int, default=1)
    parser.add_argument('--imle-temperature', action='store', type=float, default=10.0)
    parser.add_argument('--imle-lambda', action='store', type=float, default=10.0)

    # STE
    parser.add_argument('--ste-noise', type=str, choices=['none', 'sog', 'gumbel'], default='sog')
    parser.add_argument('--ste-temperature', action='store', type=float, default=0.0)

    parser.add_argument('--gradient-scaling', action='store_true', default=False)
    parser.add_argument('--seed', action='store', type=int, default=0)

    args = parser.parse_args(argv)

    batch_size = args.batch_size
    input_dim = args.input_dim
    code_m = args.code_m
    code_n = args.code_n
    nb_epochs = args.epochs

    anneal_rate = args.anneal_rate
    init_temperature = args.init_temperature
    min_temperature = args.min_temperature
    hard = args.hard

    set_seed(args.seed)

    hostname = socket.gethostname()
    logger.info(f'Hostname: {hostname}')

    wandb.init(project="aimle-dvae", name=f'{args.method}')

    wandb.config.update(args)
    wandb.config.update({'hostname': hostname, 'seed': args.seed})

    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')

    mnist_train_ds = datasets.MNIST('./mnist-data',
                                    train=True,
                                    download=True,
                                    transform=transforms.Compose([
                                        transforms.ToTensor()
                                    ]))
    train_loader = DataLoader(mnist_train_ds, batch_size=batch_size, shuffle=True)

    mnist_test_ds = datasets.MNIST('./mnist-data',
                                   train=False,
                                   download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor()
                                   ]))
    test_loader = DataLoader(mnist_test_ds, batch_size=batch_size, shuffle=True)

    model = DiscreteVAE(input_dim=input_dim, n=code_n, m=code_m).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    tau = init_temperature

    if args.aimle_target in {'standard'}:
        target_distribution = TargetDistribution(alpha=1.0,
                                                 beta=args.imle_lambda,
                                                 do_gradient_scaling=args.gradient_scaling)
    elif args.aimle_target in {'adaptive'}:
        target_distribution = AdaptiveTargetDistribution(initial_alpha=1.0,
                                                         initial_beta=args.imle_lambda,
                                                         beta_update_step=args.aimle_beta_update_step,
                                                         beta_update_momentum=args.aimle_beta_update_momentum,
                                                         target_norm=args.aimle_target_norm)
    else:
        assert False, f'Do not know how to handle {args.aimle_target} as target distribution'

    def name_to_distribution(distribution_name: str) -> Optional[BaseNoiseDistribution]:
        if distribution_name in {'none'}:
            noise_distribution = None
        elif distribution_name in {'sog'}:
            noise_distribution = SumOfGammaNoiseDistribution(k=args.select_k, nb_iterations=10, device=device)
        elif distribution_name in {'gumbel'}:
            noise_distribution = GumbelNoiseDistribution(device=device)
        else:
            assert False, f'Noise model not supported: {distribution_name}'
        return noise_distribution

    batch_idx = 0
    for epoch in range(1, nb_epochs + 1):
        epoch_loss_values = []

        # this is only needed for the standard Gumbel softmax trick
        ### tau = np.maximum(tau * np.exp(- anneal_rate * epoch), min_temperature)

        blackbox_function = lambda logits: mathias_select_k(logits, k=args.select_k)

        if args.method in {'gumbel-softmax'}:
            gumbel_softmax = lambda logits_m_2d: F.gumbel_softmax(logits_m_2d, tau=tau, hard=hard)
            code_generator = DifferentiableSelectKModel(gumbel_softmax, blackbox_function)

        elif args.method in {'imle'}:
            noise_distribution = name_to_distribution(args.imle_noise)

            @imle(target_distribution=target_distribution,
                  noise_distribution=noise_distribution,
                  nb_samples=args.imle_samples,
                  theta_noise_temperature=args.imle_temperature,
                  target_noise_temperature=args.imle_temperature)
            def imle_select_k(logits: Tensor) -> Tensor:
                return mathias_select_k(logits, k=args.select_k)

            code_generator = DifferentiableSelectKModel(imle_select_k, blackbox_function)

        elif args.method in {'aimle'}:
            noise_distribution = name_to_distribution(args.imle_noise)

            @aimle(target_distribution=target_distribution,
                   noise_distribution=noise_distribution,
                   nb_samples=args.imle_samples,
                   theta_noise_temperature=args.imle_temperature,
                   target_noise_temperature=args.imle_temperature,
                   symmetric_perturbation=args.aimle_symmetric)
            def aimle_select_k(logits: Tensor) -> Tensor:
                return mathias_select_k(logits, k=args.select_k)

            code_generator = DifferentiableSelectKModel(aimle_select_k, blackbox_function)

        elif args.method in {'ste'}:
            noise_distribution = name_to_distribution(args.ste_noise)

            @ste(noise_distribution=noise_distribution,
                 noise_temperature=args.ste_temperature,
                 nb_samples=args.imle_samples)
            def ste_select_k(logits: Tensor) -> Tensor:
                return mathias_select_k(logits, k=args.select_k)

            code_generator = DifferentiableSelectKModel(ste_select_k, blackbox_function)

        else:
            assert False, f'Unknown method: {args.method}'

        for X, _ in train_loader:
            batch_idx += 1

            if batch_idx % 1000 == 0:
                tau = np.maximum(init_temperature * np.exp(- anneal_rate * batch_idx), min_temperature)

                if args.method in {'gumbel-softmax'}:
                    gumbel_softmax = lambda logits_m_2d: F.gumbel_softmax(logits_m_2d, tau=tau, hard=hard)
                    code_generator = DifferentiableSelectKModel(gumbel_softmax, blackbox_function)

            model.train()
            X = X.to(device)

            # [B, 1, H, W]
            batch_shape = X.shape
            batch_size_ = batch_shape[0]

            assert batch_shape[1] == 1
            assert batch_shape[2] == 28
            assert batch_shape[3] == 28

            # [B, H * W]
            x_2d = X.view(batch_size_, -1)
            flat_input_size = x_2d.shape[1]

            assert x_2d.shape[0] == batch_size_
            assert x_2d.shape[1] == flat_input_size == 28 * 28

            # [B, M * N], [B * S, H * W]
            logits_2d, rec_2d = model(x_2d, code_generator=code_generator)

            nb_samples = rec_2d.shape[0] // batch_size_

            assert logits_2d.shape[0] == batch_size_
            assert logits_2d.shape[1] == code_m * code_n

            assert rec_2d.shape[0] == batch_size_ * nb_samples
            assert rec_2d.shape[1] == flat_input_size

            loss, _ = gumbel_loss(logits_2d=logits_2d, rec_2d=rec_2d, x_2d=x_2d, m=code_m, n=code_n, reduction='mean')

            loss_value = loss.item()
            epoch_loss_values += [loss_value]

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        loss_mean, loss_std = np.mean(epoch_loss_values), np.std(epoch_loss_values)
        beta_value_str = 'None' if target_distribution is None else f'{target_distribution.beta:.5f}'
        logger.info(f'Epoch {epoch}/{nb_epochs}\t'
                    f'Training Loss: {loss_mean:.4f} ± {loss_std:.4f}\t'
                    f'Temperature: {tau:.5f}\tBeta: {beta_value_str}')

        if args.method in {'gumbel-softmax'}:
            gumbel_softmax = lambda logits_m_2d: F.gumbel_softmax(logits_m_2d, tau=tau, hard=True)
            code_generator = DifferentiableSelectKModel(gumbel_softmax, blackbox_function)

        test_loss = 0.0
        nb_instances = 0.0
        with torch.inference_mode():
            for X, _ in test_loader:
                model.train()
                X = X.to(device)
                # [B, 1, H, W]
                batch_shape = X.shape
                batch_size_ = batch_shape[0]
                # [B, H * W]
                x_2d = X.view(batch_size_, -1)

                # [B, M * N], [B, H * W]
                logits_2d, rec_2d = model(x_2d, code_generator=code_generator)
                loss, nb_samples = gumbel_loss(logits_2d=logits_2d,
                                               rec_2d=rec_2d,
                                               x_2d=x_2d,
                                               m=code_m,
                                               n=code_n,
                                               reduction='sum')

                test_loss += loss.item() / nb_samples
                nb_instances += batch_size_

        logger.info(f'Average Test Loss: {test_loss / nb_instances:.5f}')

        wandb.log({'loss': loss_mean, 'tau': tau, 'test_loss': test_loss / nb_instances}, step=epoch)

    logger.info(f'Experiment completed.')


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    main(sys.argv[1:])
