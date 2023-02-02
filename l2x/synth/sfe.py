# -*- coding: utf-8 -*-

"""Score function estimator"""

import torch
from l2x.synth.utils import _maybe_ctx_call


def sfe(sampler, loss_f, grad_log_p):
    # print(f'sfe.sfe({sampler}, {loss_f}, {grad_log_p})')
    return lambda theta: _SFE.apply(theta, sampler, loss_f, grad_log_p)


# noinspection PyMethodOverriding
class _SFE(torch.autograd.Function):

    @staticmethod
    def forward(ctx, _theta, sample_strategy, loss, grad_log_p):
        # z samples
        # [10]
        ctx.sample = _maybe_ctx_call(sample_strategy, ctx, _theta)
        # θ
        ctx.theta = _theta
        # loss(z) = ((z - b_t) ** 2).sum()
        ctx.loss = _maybe_ctx_call(loss, ctx, ctx.sample)
        ctx.grad_log_p = grad_log_p
        return ctx.sample

    # Reminder: ∇θ 𝔼[ f(z) ] = 𝔼ₚ₍z;θ₎ [ f(z) ∇θ log p(z;θ) ]
    @staticmethod
    def backward(ctx, grad_output):
        return ctx.loss * _maybe_ctx_call(ctx.grad_log_p, ctx, ctx.theta), None, None, None
