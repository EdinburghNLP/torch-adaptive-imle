#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch as t
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from torch import Tensor

from imle.ste import ste as my_ste
from imle.imle import imle as my_imle
from imle.aimle import aimle as my_aimle
from imle.target import TargetDistribution, AdaptiveTargetDistribution
from imle.noise import SumOfGammaNoiseDistribution, GumbelNoiseDistribution

from aaai23.synth import distributions, utils, sfe

FIGSIZE = (3.2, 2.5)


# set the colormap and centre the colorbar
class MidpointNormalize(colors.Normalize):
    """
    Normalise the colorbar so that diverging bars work there way either side from a prescribed midpoint value)

    e.g. im=ax1.imshow(array, norm=MidpointNormalize(midpoint=0.,vmin=-100, vmax=100))
    """
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))


def optim_loop(min_objective, true_objective, lr, momentum_factor, n_steps, theta0, debug_print_loss=False):
    theta_t = t.from_numpy(theta0).float().requires_grad_(True)
    # let's try to optimize this expectation w.r.t. theta
    optimizer = t.optim.SGD([theta_t], lr, momentum=momentum_factor)
    hist, hist_expectation = [], []
    for _t in range(n_steps):
        optimizer.zero_grad()
        obj = min_objective(theta_t)
        if debug_print_loss:
            print(obj)
        hist.append(obj.detach().numpy())
        hist_expectation.append(true_objective(theta_t).detach().numpy())
        obj.backward()
        optimizer.step()
    return hist, hist_expectation


def experiment(min_obj, ture_objective, lr, theta0, momentum=0.9, steps=100, n_rp=50, do_plot=True, postprocess=None):
    # redefine objective with given strategy
    hist = []
    for _ in range(n_rp):
        # print('-', end='')
        stoc_obj, true_obj = optim_loop(min_obj, ture_objective, lr, momentum, steps, theta0)
        if postprocess:
            true_obj = postprocess(true_obj)
        hist.append(true_obj)
    if do_plot:
        mean = np.mean(hist, axis=0)
        # plt.plot(full_optim_hist)
        plt.plot(mean)
        # plt.show()

    # print()
    return hist


def plot_mean_std(histories, names, xs=None):
    means = [np.mean(np.array(his), axis=0) for his in histories]
    std_devs = [np.std(np.array(his), axis=0) for his in histories]

    for h, st, nm in zip(means, std_devs, names):
        x_axis = xs if xs else list(range(len(h)))
        line = plt.plot(xs, h, label=nm)
        plt.fill_between(x_axis, h - st, h + st, alpha=0.5, color=line[0].get_color())


def do_plots_exp(histories, names, savename=None, figsize=FIGSIZE, min_value_of_exp=None):
    # computing also standard devs
    plt.figure(figsize=figsize)
    means = [np.mean(np.array(his) - min_value_of_exp, axis=0) for his in histories]
    std_devs = [np.std(np.array(his), axis=0) for his in histories]

    for h, st, nm in zip(means, std_devs, names):
        x_axis = list(range(len(h)))
        line = plt.plot(h, label=nm)
        plt.fill_between(x_axis, h - st, h + st, alpha=0.5, color=line[0].get_color())

    plt.legend(loc=0)
    plt.ylim((0., 3.))
    plt.xlim((0, 99))
    plt.xlabel('Optimization steps')
    plt.ylabel('Optimality gap')
    if savename:
        print('Saving plots ..', savename)
        plt.savefig(savename, bbox_inches='tight')
    # plt.show()


def toy_exp_v2(n, k,
               imle_sog_hyp, imle_gum_hyp,
               aimle_sym_sog_hyp, aimle_sym_gum_hyp,
               aimle_adapt_sog_hyp, aimle_adapt_gum_hyp,
               aimle_sym_adapt_sog_hyp, aimle_sym_adapt_gum_hyp,
               ste_sog_hyp, ste_gum_hyp,
               sfe_hyp, n_rep=50):
    rng = np.random.RandomState(0)
    theta = rng.randn(n)
    topk = distributions.TopK(n, k)

    b_t = t.abs(t.from_numpy(rng.randn(n)).float())
    print(b_t)

    sorted_bt = np.sort(b_t.detach().numpy())
    min_value_of_exp = np.sum((sorted_bt[:k])**2) + np.sum((sorted_bt[k:] - 1)**2)
    print(min_value_of_exp)

    def objective(z):
        return ((z - b_t) ** 2).sum()

    # Expected value of the loss
    full_obj = lambda _th: utils.expect_obj(topk, _th, objective)

    exp = lambda strategy, lr, n_rp=n_rep, steps=100: experiment(
        lambda _th: ((strategy(_th) - b_t)**2).sum(),
        full_obj,
        lr,
        theta,
        steps=steps,
        n_rp=n_rp
    )

    # Returns a function that produces a single random scalar
    # sog_noise = utils.sum_of_gamma_noise(k, rng=np.random.RandomState(0))
    # Returns a state, so [10]
    # pam_sog = topk.perturb_and_map(sog_noise)

    # gumbel_noise = utils.gumbel_noise(rng=np.random.RandomState(0))
    # pam_gum = topk.perturb_and_map(gumbel_noise)

    # hyperparameter values obtained by grid search (sensitivity_imle)
    # Returns a f(theta) with a custom backward pass, given lambda=2.0 and sampler=pam_sog
    # imle_pid_sog = imle.imle_pid(2., pam_sog)
    # Runs the actual experiment
    # imle_pam_sog_lcs = exp(imle_pid_sog, 0.75)
    # imle_pam_gum_lcs = exp(imle.imle_pid(2., pam_gum), 0.75)

    target_distribution_sog = TargetDistribution(alpha=1.0, beta=imle_sog_hyp["lmd"], do_gradient_scaling=True)
    target_distribution_gum = TargetDistribution(alpha=1.0, beta=imle_gum_hyp["lmd"], do_gradient_scaling=True)

    sog_distribution = SumOfGammaNoiseDistribution(k=k, nb_iterations=10)
    gum_distribution = GumbelNoiseDistribution()

    @my_imle(target_distribution=target_distribution_sog, noise_distribution=sog_distribution, nb_samples=1,
             theta_noise_temperature=1.0, target_noise_temperature=1.0)
    def imle_topk_sog_batched(thetas: Tensor) -> Tensor:
        return t.stack([topk.map(thetas[i]) for i in range(thetas.shape[0])])

    def imle_topk_sog(theta: Tensor) -> Tensor:
        return imle_topk_sog_batched(theta.view(1, -1)).view(-1)

    my_imle_pam_sog_lcs = exp(imle_topk_sog, imle_sog_hyp["lr"])

    @my_imle(target_distribution=target_distribution_gum, noise_distribution=gum_distribution, nb_samples=1,
             theta_noise_temperature=1.0, target_noise_temperature=1.0)
    def imle_topk_gum_batched(thetas: Tensor) -> Tensor:
        return t.stack([topk.map(thetas[i]) for i in range(thetas.shape[0])])

    def imle_topk_gum(theta: Tensor) -> Tensor:
        return imle_topk_gum_batched(theta.view(1, -1)).view(-1)

    my_imle_pam_gum_lcs = exp(imle_topk_gum, imle_gum_hyp["lr"])

    # ---

    @my_aimle(target_distribution=target_distribution_sog, noise_distribution=sog_distribution, nb_samples=1,
              theta_noise_temperature=1.0, target_noise_temperature=1.0, symmetric_perturbation=True)
    def aimle_sym_topk_sog_batched(thetas: Tensor) -> Tensor:
        return t.stack([topk.map(thetas[i]) for i in range(thetas.shape[0])])

    def aimle_sym_topk_sog(theta: Tensor) -> Tensor:
        return imle_topk_sog_batched(theta.view(1, -1)).view(-1)

    my_aimle_sym_pam_sog_lcs = exp(aimle_sym_topk_sog, aimle_sym_sog_hyp["lr"])

    @my_aimle(target_distribution=target_distribution_gum, noise_distribution=gum_distribution, nb_samples=1,
              theta_noise_temperature=1.0, target_noise_temperature=1.0, symmetric_perturbation=True)
    def aimle_sym_topk_gum_batched(thetas: Tensor) -> Tensor:
        return t.stack([topk.map(thetas[i]) for i in range(thetas.shape[0])])

    def aimle_sym_topk_gum(theta: Tensor) -> Tensor:
        return aimle_sym_topk_gum_batched(theta.view(1, -1)).view(-1)

    my_aimle_sym_pam_gum_lcs = exp(aimle_sym_topk_gum, aimle_sym_gum_hyp["lr"])

    # ---

    target_distribution_sog_a = AdaptiveTargetDistribution(initial_beta=aimle_adapt_sog_hyp["lmd"])
    target_distribution_gum_a = AdaptiveTargetDistribution(initial_beta=aimle_adapt_gum_hyp["lmd"])

    @my_aimle(target_distribution=target_distribution_sog_a, noise_distribution=sog_distribution, nb_samples=1,
              theta_noise_temperature=1.0, target_noise_temperature=1.0, symmetric_perturbation=False)
    def aimle_adapt_topk_sog_batched(thetas: Tensor) -> Tensor:
        return t.stack([topk.map(thetas[i]) for i in range(thetas.shape[0])])

    def aimle_adapt_topk_sog(theta: Tensor) -> Tensor:
        return imle_topk_sog_batched(theta.view(1, -1)).view(-1)

    my_aimle_adapt_pam_sog_lcs = exp(aimle_adapt_topk_sog, aimle_adapt_sog_hyp["lr"])

    @my_aimle(target_distribution=target_distribution_gum_a, noise_distribution=gum_distribution, nb_samples=1,
              theta_noise_temperature=1.0, target_noise_temperature=1.0, symmetric_perturbation=False)
    def aimle_adapt_topk_gum_batched(thetas: Tensor) -> Tensor:
        return t.stack([topk.map(thetas[i]) for i in range(thetas.shape[0])])

    def aimle_adapt_topk_gum(theta: Tensor) -> Tensor:
        return aimle_sym_topk_gum_batched(theta.view(1, -1)).view(-1)

    my_aimle_adapt_pam_gum_lcs = exp(aimle_adapt_topk_gum, aimle_adapt_gum_hyp["lr"])

    # ---

    target_distribution_sog_a = AdaptiveTargetDistribution(initial_beta=aimle_sym_adapt_sog_hyp["lmd"])
    target_distribution_gum_a = AdaptiveTargetDistribution(initial_beta=aimle_sym_adapt_gum_hyp["lmd"])

    @my_aimle(target_distribution=target_distribution_sog_a, noise_distribution=sog_distribution, nb_samples=1,
              theta_noise_temperature=1.0, target_noise_temperature=1.0, symmetric_perturbation=True)
    def aimle_sym_adapt_topk_sog_batched(thetas: Tensor) -> Tensor:
        return t.stack([topk.map(thetas[i]) for i in range(thetas.shape[0])])

    def aimle_sym_adapt_topk_sog(theta: Tensor) -> Tensor:
        return imle_topk_sog_batched(theta.view(1, -1)).view(-1)

    my_aimle_sym_adapt_pam_sog_lcs = exp(aimle_sym_adapt_topk_sog, aimle_sym_adapt_sog_hyp["lr"])

    @my_aimle(target_distribution=target_distribution_gum_a, noise_distribution=gum_distribution, nb_samples=1,
              theta_noise_temperature=1.0, target_noise_temperature=1.0, symmetric_perturbation=True)
    def aimle_sym_adapt_topk_gum_batched(thetas: Tensor) -> Tensor:
        return t.stack([topk.map(thetas[i]) for i in range(thetas.shape[0])])

    def aimle_sym_adapt_topk_gum(theta: Tensor) -> Tensor:
        return aimle_sym_topk_gum_batched(theta.view(1, -1)).view(-1)

    my_aimle_sym_adapt_pam_gum_lcs = exp(aimle_sym_adapt_topk_gum, aimle_sym_adapt_gum_hyp["lr"])

    # ---

    @my_ste(noise_distribution=sog_distribution, noise_temperature=1.0)
    def ste_topk_sog_batched(thetas: Tensor) -> Tensor:
        return t.stack([topk.map(thetas[i]) for i in range(thetas.shape[0])])

    def ste_topk_sog(theta: Tensor) -> Tensor:
        return imle_topk_sog_batched(theta.view(1, -1)).view(-1)

    my_ste_pam_sog_lcs = exp(ste_topk_sog, ste_sog_hyp["lr"])

    @my_ste(noise_distribution=gum_distribution, noise_temperature=1.0)
    def ste_topk_gum_batched(thetas: Tensor) -> Tensor:
        return t.stack([topk.map(thetas[i]) for i in range(thetas.shape[0])])

    def ste_topk_gum(theta: Tensor) -> Tensor:
        return imle_topk_gum_batched(theta.view(1, -1)).view(-1)

    my_ste_pam_gum_lcs = exp(ste_topk_gum, ste_gum_hyp["lr"])

    # ---

    # hyperparameter values obtained by grid search (sensitivity_ste)
    # ste_pam_lcs = exp(ste.ste(pam_gum), 0.019)

    do_plots_exp(
        [
            # my_ste_pam_sog_lcs,
            my_ste_pam_gum_lcs,

            # my_imle_pam_sog_lcs,
            my_imle_pam_gum_lcs,

            # my_aimle_sym_pam_sog_lcs,
            my_aimle_sym_pam_gum_lcs,

            # my_aimle_adapt_pam_sog_lcs,
            my_aimle_adapt_pam_gum_lcs,

            # my_aimle_sym_adapt_pam_sog_lcs,
            my_aimle_sym_adapt_pam_gum_lcs
        ], [
            # 'STE PaM (SoG)',
            'STE PaM (Gum)',

            # 'I-MLE PaM (SoG)',
            'I-MLE PaM (Gum)',

            # 'AI-MLE Sym PaM (SoG)',
            'AI-MLE Sym PaM (Gum)',

            # 'AI-MLE Adapt PaM (SoG)',
            'AI-MLE Adapt PaM (Gum)',

            # 'AI-MLE Sym Adapt PaM (SoG)',
            'AI-MLE Sym Adapt PaM (Gum)'
        ], savename='SYNTH_v2.pdf', figsize=(4, 3), min_value_of_exp=min_value_of_exp)

    # hyperparameter values obtained by grid search (sensitivity_sfe)
    sfe_full = sfe.sfe(topk.sample_f(np.random.RandomState(0)), objective, topk.grad_log_p(topk.marginals))
    sfe_full_lcs = exp(sfe_full, sfe_hyp["lr"], n_rp=n_rep//5, steps=1000)
    ary_sfe = np.array(sfe_full_lcs)

    # final plot!
    do_plots_exp(
        [
            ary_sfe[:, ::10],

            # my_ste_pam_sog_lcs,
            my_ste_pam_gum_lcs,

            # my_imle_pam_sog_lcs,
            my_imle_pam_gum_lcs,

            # my_aimle_sym_pam_sog_lcs,
            my_aimle_sym_pam_gum_lcs,

            # my_aimle_adapt_pam_sog_lcs,
            my_aimle_adapt_pam_gum_lcs,

            # my_aimle_sym_adapt_pam_sog_lcs,
            my_aimle_sym_adapt_pam_gum_lcs
        ], [
            'SFE (steps x 10)',

            # 'STE PaM (SoG)',
            'STE PaM (Gum)',

            # 'I-MLE PaM (SoG)',
            'I-MLE PaM (Gum)',

            # 'AI-MLE PaM (SoG)',
            'AI-MLE PaM (Gum)',

            # 'AI-MLE Adapt PaM (SoG)',
            'AI-MLE Adapt PaM (Gum)'
        ], savename='SYNTH2.pdf', min_value_of_exp=min_value_of_exp)


def sensibility_imle_v2(n, k, n_rep=20):
    rng = np.random.RandomState(0)
    theta = rng.randn(n)
    topk = distributions.TopK(n, k)

    b_t = t.abs(t.from_numpy(rng.randn(n)).float())
    # print(b_t)

    sorted_bt = np.sort(b_t.detach().numpy())
    min_value_of_exp = np.sum((sorted_bt[:k])**2) + np.sum((sorted_bt[k:] - 1)**2)
    # print(min_value_of_exp)

    def objective(z):
        return ((z - b_t)**2).sum()

    full_obj = lambda _th: utils.expect_obj(topk, _th, objective)

    def pp(_his):
        if _his[-1] - min_value_of_exp < 0.:  # then it's all lost
            _his[-1] = 5.
            # print('pp')
        return _his

    exp = lambda strategy, lr, n_rp=n_rep, steps=100: experiment(
        lambda _th: ((strategy(_th) - b_t)**2).sum(),
        full_obj,
        lr, theta, steps=steps, n_rp=n_rp, do_plot=False,
        postprocess=pp
    )

    n_lr, n_lbd = 5, 6

    search_grid_lr = np.linspace(0.5, 1., num=n_lr)
    search_grid_lambda = np.linspace(0.5, 3., num=n_lbd)

    res_sog_mean, res_sog_std = np.zeros((n_lr, n_lbd)), np.zeros((n_lr, n_lbd))
    res_gum_mean, res_gum_std = np.zeros((n_lr, n_lbd)), np.zeros((n_lr, n_lbd))

    best_sog, sog_hyp = None, None
    best_gum, gum_hyp = None, None

    for i, lr in enumerate(search_grid_lr):
        for j, lmd in enumerate(search_grid_lambda):
            # print(i, j)

            # pam_sog = topk.perturb_and_map(utils.sum_of_gamma_noise(k, rng=np.random.RandomState(0)))
            # pam_gum = topk.perturb_and_map(utils.gumbel_noise(rng=np.random.RandomState(0)))

            # imle_sog_lcs = exp(imle.imle_pid(lmd, pam_sog), lr)
            # imle_gum_lcs = exp(imle.imle_pid(lmd, pam_gum), lr)

            target_distribution = TargetDistribution(alpha=1.0, beta=lmd, do_gradient_scaling=True)
            sog_distribution = SumOfGammaNoiseDistribution(k=k, nb_iterations=10)
            gum_distribution = GumbelNoiseDistribution()

            @my_imle(target_distribution=target_distribution, noise_distribution=sog_distribution, nb_samples=1,
                     theta_noise_temperature=1.0, target_noise_temperature=1.0)
            def imle_topk_sog_batched(thetas: Tensor) -> Tensor:
                return t.stack([topk.map(thetas[i]) for i in range(thetas.shape[0])])

            def imle_topk_sog(theta: Tensor) -> Tensor:
                return imle_topk_sog_batched(theta.view(1, -1)).view(-1)

            @my_imle(target_distribution=target_distribution, noise_distribution=gum_distribution, nb_samples=1,
                     theta_noise_temperature=1.0, target_noise_temperature=1.0)
            def imle_topk_gum_batched(thetas: Tensor) -> Tensor:
                return t.stack([topk.map(thetas[i]) for i in range(thetas.shape[0])])

            def imle_topk_gum(theta: Tensor) -> Tensor:
                return imle_topk_sog_batched(theta.view(1, -1)).view(-1)

            imle_sog_lcs = exp(imle_topk_sog, lr)
            imle_gum_lcs = exp(imle_topk_gum, lr)

            res_sog_mean[i, j] = np.mean(np.array(imle_sog_lcs) - min_value_of_exp, axis=0)[-1]
            res_sog_std[i, j] = np.std(np.array(imle_sog_lcs), axis=0)[-1]

            if best_sog is None or res_sog_mean[i, j] < best_sog:
                best_sog = res_sog_mean[i, j]
                sog_hyp = {"lr": lr, "lmd": lmd, "loss": best_sog}

            res_gum_mean[i, j] = np.mean(np.array(imle_gum_lcs) - min_value_of_exp, axis=0)[-1]
            res_gum_std[i, j] = np.std(np.array(imle_gum_lcs), axis=0)[-1]

            if best_gum is None or res_gum_mean[i, j] < best_gum:
                best_gum = res_gum_mean[i, j]
                gum_hyp = {"lr": lr, "lmd": lmd, "loss": best_gum}

            print(f'XXX lr: {lr} lmd: {lmd} res_sog {res_sog_mean[i, j]:.5f} res_gum {res_gum_mean[i, j]:.5f}')

    return sog_hyp, gum_hyp


def sensibility_aimle_v2(n, k, is_sym, n_rep=20):
    rng = np.random.RandomState(0)
    theta = rng.randn(n)
    topk = distributions.TopK(n, k)

    b_t = t.abs(t.from_numpy(rng.randn(n)).float())
    # print(b_t)

    sorted_bt = np.sort(b_t.detach().numpy())
    min_value_of_exp = np.sum((sorted_bt[:k])**2) + np.sum((sorted_bt[k:] - 1)**2)
    # print(min_value_of_exp)

    def objective(z):
        return ((z - b_t)**2).sum()

    full_obj = lambda _th: utils.expect_obj(topk, _th, objective)

    def pp(_his):
        if _his[-1] - min_value_of_exp < 0.:  # then it's all lost
            _his[-1] = 5.
            # print('pp')
        return _his

    exp = lambda strategy, lr, n_rp=n_rep, steps=100: experiment(
        lambda _th: ((strategy(_th) - b_t)**2).sum(),
        full_obj,
        lr, theta, steps=steps, n_rp=n_rp, do_plot=False,
        postprocess=pp
    )

    n_lr, n_lbd = 5, 6

    search_grid_lr = np.linspace(0.5, 1., num=n_lr)
    search_grid_lambda = np.linspace(0.5, 3., num=n_lbd)

    res_sog_mean, res_sog_std = np.zeros((n_lr, n_lbd)), np.zeros((n_lr, n_lbd))
    res_gum_mean, res_gum_std = np.zeros((n_lr, n_lbd)), np.zeros((n_lr, n_lbd))

    best_sog, sog_hyp = None, None
    best_gum, gum_hyp = None, None

    for i, lr in enumerate(search_grid_lr):
        for j, lmd in enumerate(search_grid_lambda):
            # print(i, j)

            # pam_sog = topk.perturb_and_map(utils.sum_of_gamma_noise(k, rng=np.random.RandomState(0)))
            # pam_gum = topk.perturb_and_map(utils.gumbel_noise(rng=np.random.RandomState(0)))

            # imle_sog_lcs = exp(imle.imle_pid(lmd, pam_sog), lr)
            # imle_gum_lcs = exp(imle.imle_pid(lmd, pam_gum), lr)

            target_distribution = TargetDistribution(alpha=1.0, beta=lmd, do_gradient_scaling=True)
            sog_distribution = SumOfGammaNoiseDistribution(k=k, nb_iterations=10)
            gum_distribution = GumbelNoiseDistribution()

            @my_aimle(target_distribution=target_distribution, noise_distribution=sog_distribution, nb_samples=1,
                      theta_noise_temperature=1.0, target_noise_temperature=1.0, symmetric_perturbation=is_sym)
            def imle_topk_sog_batched(thetas: Tensor) -> Tensor:
                return t.stack([topk.map(thetas[i]) for i in range(thetas.shape[0])])

            def imle_topk_sog(theta: Tensor) -> Tensor:
                return imle_topk_sog_batched(theta.view(1, -1)).view(-1)

            @my_aimle(target_distribution=target_distribution, noise_distribution=gum_distribution, nb_samples=1,
                      theta_noise_temperature=1.0, target_noise_temperature=1.0, symmetric_perturbation=is_sym)
            def imle_topk_gum_batched(thetas: Tensor) -> Tensor:
                return t.stack([topk.map(thetas[i]) for i in range(thetas.shape[0])])

            def imle_topk_gum(theta: Tensor) -> Tensor:
                return imle_topk_sog_batched(theta.view(1, -1)).view(-1)

            imle_sog_lcs = exp(imle_topk_sog, lr)
            imle_gum_lcs = exp(imle_topk_gum, lr)

            res_sog_mean[i, j] = np.mean(np.array(imle_sog_lcs) - min_value_of_exp, axis=0)[-1]
            res_sog_std[i, j] = np.std(np.array(imle_sog_lcs), axis=0)[-1]

            if best_sog is None or res_sog_mean[i, j] < best_sog:
                best_sog = res_sog_mean[i, j]
                sog_hyp = {"lr": lr, "lmd": lmd, "loss": best_sog}

            res_gum_mean[i, j] = np.mean(np.array(imle_gum_lcs) - min_value_of_exp, axis=0)[-1]
            res_gum_std[i, j] = np.std(np.array(imle_gum_lcs), axis=0)[-1]

            if best_gum is None or res_gum_mean[i, j] < best_gum:
                best_gum = res_gum_mean[i, j]
                gum_hyp = {"lr": lr, "lmd": lmd, "loss": best_gum}

            print(f'XXX lr: {lr} lmd: {lmd} res_sog {res_sog_mean[i, j]:.5f} res_gum {res_gum_mean[i, j]:.5f}')

    return sog_hyp, gum_hyp


def sensibility_aimle_adapt_v2(n, k, n_rep=20):
    rng = np.random.RandomState(0)
    theta = rng.randn(n)
    topk = distributions.TopK(n, k)

    b_t = t.abs(t.from_numpy(rng.randn(n)).float())
    # print(b_t)

    sorted_bt = np.sort(b_t.detach().numpy())
    min_value_of_exp = np.sum((sorted_bt[:k])**2) + np.sum((sorted_bt[k:] - 1)**2)
    # print(min_value_of_exp)

    def objective(z):
        return ((z - b_t)**2).sum()

    full_obj = lambda _th: utils.expect_obj(topk, _th, objective)

    def pp(_his):
        if _his[-1] - min_value_of_exp < 0.:  # then it's all lost
            _his[-1] = 5.
            # print('pp')
        return _his

    exp = lambda strategy, lr, n_rp=n_rep, steps=100: experiment(
        lambda _th: ((strategy(_th) - b_t)**2).sum(),
        full_obj,
        lr, theta, steps=steps, n_rp=n_rp, do_plot=False,
        postprocess=pp
    )

    n_lr, n_lbd = 5, 1

    search_grid_lr = np.linspace(0.5, 1., num=n_lr)
    search_grid_lambda = np.array([0.0]) # np.linspace(0.5, 3., num=n_lbd)

    res_sog_mean, res_sog_std = np.zeros((n_lr, n_lbd)), np.zeros((n_lr, n_lbd))
    res_gum_mean, res_gum_std = np.zeros((n_lr, n_lbd)), np.zeros((n_lr, n_lbd))

    best_sog, sog_hyp = None, None
    best_gum, gum_hyp = None, None

    for i, lr in enumerate(search_grid_lr):
        for j, lmd in enumerate(search_grid_lambda):
            # print(i, j)

            # pam_sog = topk.perturb_and_map(utils.sum_of_gamma_noise(k, rng=np.random.RandomState(0)))
            # pam_gum = topk.perturb_and_map(utils.gumbel_noise(rng=np.random.RandomState(0)))

            # imle_sog_lcs = exp(imle.imle_pid(lmd, pam_sog), lr)
            # imle_gum_lcs = exp(imle.imle_pid(lmd, pam_gum), lr)

            target_distribution = AdaptiveTargetDistribution(initial_beta=lmd)
            sog_distribution = SumOfGammaNoiseDistribution(k=k, nb_iterations=10)
            gum_distribution = GumbelNoiseDistribution()

            @my_aimle(target_distribution=target_distribution, noise_distribution=sog_distribution, nb_samples=1,
                      theta_noise_temperature=1.0, target_noise_temperature=1.0, symmetric_perturbation=False)
            def imle_topk_sog_batched(thetas: Tensor) -> Tensor:
                return t.stack([topk.map(thetas[i]) for i in range(thetas.shape[0])])

            def imle_topk_sog(theta: Tensor) -> Tensor:
                return imle_topk_sog_batched(theta.view(1, -1)).view(-1)

            @my_aimle(target_distribution=target_distribution, noise_distribution=gum_distribution, nb_samples=1,
                      theta_noise_temperature=1.0, target_noise_temperature=1.0, symmetric_perturbation=False)
            def imle_topk_gum_batched(thetas: Tensor) -> Tensor:
                return t.stack([topk.map(thetas[i]) for i in range(thetas.shape[0])])

            def imle_topk_gum(theta: Tensor) -> Tensor:
                return imle_topk_gum_batched(theta.view(1, -1)).view(-1)

            imle_sog_lcs = exp(imle_topk_sog, lr)
            imle_gum_lcs = exp(imle_topk_gum, lr)

            res_sog_mean[i, j] = np.mean(np.array(imle_sog_lcs) - min_value_of_exp, axis=0)[-1]
            res_sog_std[i, j] = np.std(np.array(imle_sog_lcs), axis=0)[-1]

            if best_sog is None or res_sog_mean[i, j] < best_sog:
                best_sog = res_sog_mean[i, j]
                sog_hyp = {"lr": lr, "lmd": lmd, "loss": best_sog}

            res_gum_mean[i, j] = np.mean(np.array(imle_gum_lcs) - min_value_of_exp, axis=0)[-1]
            res_gum_std[i, j] = np.std(np.array(imle_gum_lcs), axis=0)[-1]

            if best_gum is None or res_gum_mean[i, j] < best_gum:
                best_gum = res_gum_mean[i, j]
                gum_hyp = {"lr": lr, "lmd": lmd, "loss": best_gum}

            print(f'XXX lr: {lr} lmd: {lmd} res_sog {res_sog_mean[i, j]:.5f} res_gum {res_gum_mean[i, j]:.5f}')

    return sog_hyp, gum_hyp


def sensibility_aimle_adapt_v2(n, k, is_sym, n_rep=20):
    rng = np.random.RandomState(0)
    theta = rng.randn(n)
    topk = distributions.TopK(n, k)

    b_t = t.abs(t.from_numpy(rng.randn(n)).float())

    sorted_bt = np.sort(b_t.detach().numpy())
    min_value_of_exp = np.sum((sorted_bt[:k])**2) + np.sum((sorted_bt[k:] - 1)**2)

    def objective(z):
        return ((z - b_t)**2).sum()

    full_obj = lambda _th: utils.expect_obj(topk, _th, objective)

    def pp(_his):
        if _his[-1] - min_value_of_exp < 0.:  # then it's all lost
            _his[-1] = 5.
            # print('pp')
        return _his

    exp = lambda strategy, lr, n_rp=n_rep, steps=100: experiment(
        lambda _th: ((strategy(_th) - b_t)**2).sum(),
        full_obj,
        lr, theta, steps=steps, n_rp=n_rp, do_plot=False,
        postprocess=pp
    )

    n_lr, n_lbd = 5, 1

    search_grid_lr = np.linspace(0.5, 1., num=n_lr)
    search_grid_lambda = np.array([0.0]) # np.linspace(0.5, 3., num=n_lbd)

    res_sog_mean, res_sog_std = np.zeros((n_lr, n_lbd)), np.zeros((n_lr, n_lbd))
    res_gum_mean, res_gum_std = np.zeros((n_lr, n_lbd)), np.zeros((n_lr, n_lbd))

    best_sog, sog_hyp = None, None
    best_gum, gum_hyp = None, None

    for i, lr in enumerate(search_grid_lr):
        for j, lmd in enumerate(search_grid_lambda):
            # print(i, j)

            # pam_sog = topk.perturb_and_map(utils.sum_of_gamma_noise(k, rng=np.random.RandomState(0)))
            # pam_gum = topk.perturb_and_map(utils.gumbel_noise(rng=np.random.RandomState(0)))

            # imle_sog_lcs = exp(imle.imle_pid(lmd, pam_sog), lr)
            # imle_gum_lcs = exp(imle.imle_pid(lmd, pam_gum), lr)

            target_distribution = AdaptiveTargetDistribution(initial_beta=lmd)
            sog_distribution = SumOfGammaNoiseDistribution(k=k, nb_iterations=10)
            gum_distribution = GumbelNoiseDistribution()

            @my_aimle(target_distribution=target_distribution, noise_distribution=sog_distribution, nb_samples=1,
                      theta_noise_temperature=1.0, target_noise_temperature=1.0, symmetric_perturbation=is_sym)
            def imle_topk_sog_batched(thetas: Tensor) -> Tensor:
                return t.stack([topk.map(thetas[i]) for i in range(thetas.shape[0])])

            def imle_topk_sog(theta: Tensor) -> Tensor:
                return imle_topk_sog_batched(theta.view(1, -1)).view(-1)

            @my_aimle(target_distribution=target_distribution, noise_distribution=gum_distribution, nb_samples=1,
                      theta_noise_temperature=1.0, target_noise_temperature=1.0, symmetric_perturbation=is_sym)
            def imle_topk_gum_batched(thetas: Tensor) -> Tensor:
                return t.stack([topk.map(thetas[i]) for i in range(thetas.shape[0])])

            def imle_topk_gum(theta: Tensor) -> Tensor:
                return imle_topk_sog_batched(theta.view(1, -1)).view(-1)

            imle_sog_lcs = exp(imle_topk_sog, lr)
            imle_gum_lcs = exp(imle_topk_gum, lr)

            res_sog_mean[i, j] = np.mean(np.array(imle_sog_lcs) - min_value_of_exp, axis=0)[-1]
            res_sog_std[i, j] = np.std(np.array(imle_sog_lcs), axis=0)[-1]

            if best_sog is None or res_sog_mean[i, j] < best_sog:
                best_sog = res_sog_mean[i, j]
                sog_hyp = {"lr": lr, "lmd": lmd, "loss": best_sog}

            res_gum_mean[i, j] = np.mean(np.array(imle_gum_lcs) - min_value_of_exp, axis=0)[-1]
            res_gum_std[i, j] = np.std(np.array(imle_gum_lcs), axis=0)[-1]

            if best_gum is None or res_gum_mean[i, j] < best_gum:
                best_gum = res_gum_mean[i, j]
                gum_hyp = {"lr": lr, "lmd": lmd, "loss": best_gum}

            print(f'XXX lr: {lr} lmd: {lmd} res_sog {res_sog_mean[i, j]:.5f} res_gum {res_gum_mean[i, j]:.5f}')

    return sog_hyp, gum_hyp


def sensibility_ste_v2(n, k, n_rep=20):
    rng = np.random.RandomState(0)
    theta = rng.randn(n)
    topk = distributions.TopK(n, k)

    b_t = t.abs(t.from_numpy(rng.randn(n)).float())
    # print(b_t)

    sorted_bt = np.sort(b_t.detach().numpy())
    min_value_of_exp = np.sum((sorted_bt[:k])**2) + np.sum((sorted_bt[k:] - 1)**2)
    # print(min_value_of_exp)

    def objective(z):
        return ((z - b_t)**2).sum()

    full_obj = lambda _th: utils.expect_obj(topk, _th, objective)

    def pp(_his):
        if _his[-1] - min_value_of_exp < 0.:  # then it's all lost
            _his[-1] = 5.
            # print('pp')
        return _his

    exp = lambda strategy, lr, n_rp=n_rep, steps=100: experiment(
        lambda _th: ((strategy(_th) - b_t)**2).sum(),
        full_obj,
        lr, theta, steps=steps, n_rp=n_rp, do_plot=False,
        postprocess=pp
    )

    n_lr, n_lbd = 10, 1

    search_grid_lr = np.exp(np.linspace(np.log(0.001), np.log(.2), num=n_lr))
    search_grid_lambda = np.linspace(0.5, 3., num=n_lbd)

    res_ste_mean, res_ste_std = np.zeros((n_lr, n_lbd)), np.zeros((n_lr, n_lbd))
    res_ste_g_mean, res_ste_g_std = np.zeros((n_lr, n_lbd)), np.zeros((n_lr, n_lbd))
    # res_gum_mean, res_gum_std = np.zeros((n_lr, n_lbd)), np.zeros((n_lr, n_lbd))

    best_sog, sog_hyp = None, None
    best_gum, gum_hyp = None, None

    for i, lr in enumerate(search_grid_lr):
        for j, lmd in enumerate(search_grid_lambda):
            # print(i, j)
            #pam_sog = topk.perturb_and_map(utils.sum_of_gamma_noise(k, rng=np.random.RandomState(0)))
            #pam_gum = topk.perturb_and_map(utils.gumbel_noise(rng=np.random.RandomState(0)))

            #ste_pam_lcs = exp(ste.ste(pam_sog), lr)
            #ste_pam_g_lcs = exp(ste.ste(pam_gum), lr)

            sog_distribution = SumOfGammaNoiseDistribution(k=k, nb_iterations=10)
            gum_distribution = GumbelNoiseDistribution()

            @my_ste(noise_distribution=sog_distribution, noise_temperature=1.0)
            def ste_topk_sog_batched(thetas: Tensor) -> Tensor:
                return t.stack([topk.map(thetas[i]) for i in range(thetas.shape[0])])

            def ste_topk_sog(theta: Tensor) -> Tensor:
                return ste_topk_sog_batched(theta.view(1, -1)).view(-1)

            @my_ste(noise_distribution=gum_distribution, noise_temperature=1.0)
            def ste_topk_gum_batched(thetas: Tensor) -> Tensor:
                return t.stack([topk.map(thetas[i]) for i in range(thetas.shape[0])])

            def ste_topk_gum(theta: Tensor) -> Tensor:
                return ste_topk_sog_batched(theta.view(1, -1)).view(-1)

            ste_pam_lcs = exp(ste_topk_sog, lr)
            ste_pam_g_lcs = exp(ste_topk_gum, lr)

            res_ste_mean[i, j] = np.mean(np.array(ste_pam_lcs) - min_value_of_exp, axis=0)[-1]
            # res_sog_std[i, j] = np.std(np.array(imle_sog_lcs), axis=0)[-1]

            if best_sog is None or res_ste_mean[i, j] < best_sog:
                best_sog = res_ste_mean[i, j]
                sog_hyp = {"lr": lr, "lmd": lmd, "loss": best_sog}

            res_ste_g_mean[i, j] = np.mean(np.array(ste_pam_g_lcs) - min_value_of_exp, axis=0)[-1]
            # res_gum_std[i, j] = np.std(np.array(imle_gum_lcs), axis=0)[-1]

            if best_gum is None or res_ste_g_mean[i, j] < best_gum:
                best_gum = res_ste_g_mean[i, j]
                gum_hyp = {"lr": lr, "lmd": lmd, "loss": best_gum}

            print(f'XXX lr {lr:.5f} lmd {lmd} ste_mean {res_ste_mean[i, j]:.5f} ste_g_mean {res_ste_g_mean[i, j]:.5f}')

    return sog_hyp, gum_hyp


def sensibility_sfe(n, k, n_rep=20):
    rng = np.random.RandomState(0)
    theta = rng.randn(n)
    topk = distributions.TopK(n, k)

    b_t = t.abs(t.from_numpy(rng.randn(n)).float())
    print(b_t)

    sorted_bt = np.sort(b_t.detach().numpy())
    min_value_of_exp = np.sum((sorted_bt[:k])**2) + np.sum((sorted_bt[k:] - 1)**2)
    print(min_value_of_exp)

    def objective(z):
        return ((z - b_t)**2).sum()

    full_obj = lambda _th: utils.expect_obj(topk, _th, objective)

    def pp(_his):
        if _his[-1] - min_value_of_exp < 0.:  # then it's all lost
            _his[-1] = 5.
            # print('pp')
        return _his

    exp = lambda strategy, lr, n_rp=n_rep, steps=100: experiment(
        lambda _th: ((strategy(_th) - b_t)**2).sum(),
        full_obj,
        lr, theta, steps=steps, n_rp=n_rp, do_plot=False,
        postprocess=pp
    )

    n_lr, n_lbd = 10, 1

    search_grid_lr = np.exp(np.linspace(np.log(0.0001), np.log(.1), num=n_lr))
    search_grid_lambda = np.linspace(0.5, 3., num=n_lbd)

    res_sfe_mean, res_sfe_std = np.zeros((n_lr, n_lbd)), np.zeros((n_lr, n_lbd))

    best, hyp = None, None

    for i, lr in enumerate(search_grid_lr):
        for j, lmd in enumerate(search_grid_lambda):
            print(i, j)
            sfe_full = sfe.sfe(topk.sample_f(np.random.RandomState(0)),
                               objective, topk.grad_log_p(topk.marginals))
            sfe_full_lcs = exp(sfe_full, lr, n_rp=n_rep, steps=1000)

            res_sfe_mean[i, j] = np.mean(np.array(sfe_full_lcs) - min_value_of_exp, axis=0)[-1]
            res_sfe_std[i, j] = np.std(np.array(sfe_full_lcs), axis=0)[-1]

            if best is None or res_sfe_mean[i, j] < best:
                best = res_sfe_mean[i, j]
                hyp = {"lr": lr, "lmd": lmd, "loss": best}

    return hyp


if __name__ == '__main__':
    def start_process():
        pass

    def sensibility(model):
        if model in {'imle'}:
            res = sensibility_imle_v2(10, 5, n_rep=100)
            print('Best IMLE hyp:', res)
        elif model in {'aimle_sym'}:
            res = sensibility_aimle_v2(10, 5, is_sym=True, n_rep=100)
            print('Best AIMLE sym hyp:', res)
        elif model in {'aimle_adapt'}:
            res = sensibility_aimle_adapt_v2(10, 5, is_sym=False, n_rep=100)
            print('Best AIMLE adapt hyp:', res)
        elif model in {'aimle_sym_adapt'}:
            res = sensibility_aimle_adapt_v2(10, 5, is_sym=True, n_rep=100)
            print('Best AIMLE sym adapt hyp:', res)
        elif model in {'ste'}:
            res = sensibility_ste_v2(10, 5, n_rep=100)
            print('Best STE hyp:', res)
        elif model in {'sfe'}:
            res = sensibility_sfe(10, 5, n_rep=100)
            print('Best SFE hyp:', res)
        else:
            assert False, f'{model} not supported'
        return res

    # imle_sog_hyp, imle_gum_hyp = sensibility_imle_v2(10, 5, n_rep=100)
    # print('Best IMLE hyp:', imle_sog_hyp, imle_gum_hyp)

    # aimle_sym_sog_hyp, aimle_sym_gum_hyp = sensibility_aimle_sym_v2(10, 5, n_rep=100)
    # print('Best AIMLE sym hyp:', aimle_sym_sog_hyp, aimle_sym_gum_hyp)

    # ste_sog_hyp, ste_gum_hyp = sensibility_ste_v2(10, 5, n_rep=100)
    # print('Best STE hyp:', ste_sog_hyp, ste_gum_hyp)

    # sfe_hyp = sensibility_sfe(10, 5, n_rep=100)
    # print('Best SFE hyp:', sfe_hyp)

    import os
    import json
    from multiprocessing.pool import ThreadPool

    cache_path = 'synth_cache.json'

    cache = dict()
    if os.path.exists(cache_path):
        with open(cache_path, 'r') as f:
            cache = json.load(f)

    pool = ThreadPool(processes=32, initializer=start_process)

    all_keys = ['imle', 'aimle_sym', 'ste', 'sfe', 'aimle_adapt', 'aimle_sym_adapt']
    missing_keys = sorted({k for k in all_keys} - {k for k in cache})

    hyp_lst = pool.map(sensibility, missing_keys)

    for key, hyp in zip(missing_keys, hyp_lst):
        cache[key] = hyp

    with open(cache_path, 'w') as f:
        json.dump(cache, f)

    imle_sog_hyp, imle_gum_hyp = cache['imle']
    aimle_sym_sog_hyp, aimle_sym_gum_hyp = cache['aimle_sym']
    ste_sog_hyp, ste_gum_hyp = cache['ste']
    sfe_hyp = cache['sfe']
    aimle_adapt_sog_hyp, aimle_adapt_gum_hyp = cache['aimle_adapt']
    aimle_sym_adapt_sog_hyp, aimle_sym_adapt_gum_hyp = cache['aimle_sym_adapt']

    # toy_exp(10, 5, n_rep=100, an='_final')
    toy_exp_v2(10, 5, n_rep=100,
               imle_sog_hyp=imle_sog_hyp, imle_gum_hyp=imle_gum_hyp,
               aimle_sym_sog_hyp=aimle_sym_sog_hyp, aimle_sym_gum_hyp=aimle_sym_gum_hyp,
               aimle_adapt_sog_hyp=aimle_adapt_sog_hyp, aimle_adapt_gum_hyp=aimle_adapt_gum_hyp,
               aimle_sym_adapt_sog_hyp=aimle_adapt_sog_hyp, aimle_sym_adapt_gum_hyp=aimle_adapt_gum_hyp,
               ste_sog_hyp=ste_sog_hyp, ste_gum_hyp=ste_gum_hyp,
               sfe_hyp=sfe_hyp)
