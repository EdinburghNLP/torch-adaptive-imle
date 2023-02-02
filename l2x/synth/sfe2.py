# -*- coding: utf-8 -*-

"""Score function estimator"""

import torch
from l2x.synth.utils import _maybe_ctx_call


def sfe(sampler, loss_f, grad_log_p, nb_samples):
    # print(f'sfe2.sfe({sampler}, {loss_f}, {grad_log_p})')
    return lambda theta: _SFE.apply(theta, sampler, loss_f, grad_log_p, nb_samples)


# noinspection PyMethodOverriding
class _SFE(torch.autograd.Function):

    @staticmethod
    def forward(ctx, _theta, sample_strategy, loss, grad_log_p, nb_samples):
        # z samples
        ctx.nb_samples = nb_samples
        if ctx.nb_samples is None:
            ctx.sample = _maybe_ctx_call(sample_strategy, ctx, _theta)
        else:
            ctx.sample = [_maybe_ctx_call(sample_strategy, ctx, _theta) for _ in range(nb_samples)]
        # θ
        ctx.theta = _theta
        # loss(z) = ((z - b_t)**2).sum()
        if ctx.nb_samples is None:
            ctx.loss = _maybe_ctx_call(loss, ctx, ctx.sample)
        else:
            ctx.loss = [_maybe_ctx_call(loss, ctx, s) for s in ctx.sample]
        ctx.grad_log_p = grad_log_p
        res = ctx.sample
        if ctx.nb_samples is not None:
            res = torch.stack(res, dim=0)
        return res

    # Reminder: ∇θ 𝔼[ f(z) ] = 𝔼ₚ₍z;θ₎ [ f(z) ∇θ log p(z;θ) ]
    @staticmethod
    def backward(ctx, grad_output):
        if ctx.nb_samples is None:
            grad = ctx.loss * _maybe_ctx_call(ctx.grad_log_p, ctx, ctx.theta)
        else:
            grad = 0.0
            all_samples = ctx.sample
            for i, sample in enumerate(all_samples):
                ctx.sample = sample
                local_grad = ctx.loss[i] * _maybe_ctx_call(ctx.grad_log_p, ctx, ctx.theta)
                grad = grad + local_grad
            ctx.sample = all_samples
            grad = grad / ctx.nb_samples
        return grad, None, None, None, None
