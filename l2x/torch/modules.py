# -*- coding: utf-8 -*-

import torch
from torch import nn, Tensor
from torch.distributions.gamma import Gamma

from torch.distributions import Uniform

import math

from typing import Optional, Tuple, Callable

import logging

logger = logging.getLogger(__name__)


def init(layer: nn.Module):
    if isinstance(layer, nn.Conv1d) or isinstance(layer, nn.Linear):
        torch.nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain('relu'))
        torch.nn.init.zeros_(layer.bias)
    else:
        assert f'Do not know how to deal with {type(layer)}'


class GumbelSelector(torch.nn.Module):
    def __init__(self,
                 embedding_weights: Tensor,
                 kernel_size: int):
        super().__init__()

        self.nb_words = embedding_weights.shape[0]
        self.embedding_dim = embedding_weights.shape[1]

        self.embeddings = nn.Embedding.from_pretrained(embedding_weights, freeze=True)

        self.first_layer = nn.Conv1d(in_channels=self.embedding_dim, out_channels=100,
                                     kernel_size=(kernel_size,), padding='same', stride=(1,))
        init(self.first_layer)

        self.global_layer = nn.Linear(in_features=100, out_features=100, bias=True)
        init(self.global_layer)

        self.local_layer_1 = nn.Conv1d(in_channels=100, out_channels=100,
                                       kernel_size=(kernel_size,), padding='same', stride=(1,))
        init(self.local_layer_1)

        self.local_layer_2 = nn.Conv1d(in_channels=100, out_channels=100,
                                       kernel_size=(kernel_size,), padding='same', stride=(1,))
        init(self.local_layer_2)

        self.final_layer_1 = nn.Conv1d(in_channels=200, out_channels=100,
                                       kernel_size=(1,), padding='same', stride=(1,))
        init(self.final_layer_1)

        self.final_layer_2 = nn.Conv1d(in_channels=100, out_channels=1,
                                       kernel_size=(1,), padding='same', stride=(1,))
        init(self.final_layer_2)

        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)

    def forward(self,
                x: Tensor) -> Tensor:
        # [B, T] -> [B, T, E]
        x_emb = self.embeddings(x)

        batch_size = x_emb.shape[0]
        seq_len = x_emb.shape[1]
        emb_size = x_emb.shape[2]

        # [B, E, T]
        x_emb = torch.transpose(x_emb, 1, 2)  # x_emb
        assert x_emb.shape == torch.Size([batch_size, emb_size, seq_len])

        # [B, 100, T]
        first_rep = self.first_layer(x_emb)
        first_rep = self.activation(first_rep)  # A
        hidden_size = first_rep.shape[1]  # 100
        assert first_rep.shape == torch.Size([batch_size, hidden_size, seq_len])

        # [B, 100]
        global_info, _ = torch.max(input=first_rep, dim=2)  # B
        global_info = self.global_layer(global_info)
        global_info = self.activation(global_info)  # C
        # print('[B, 100]', global_info.shape)
        assert global_info.shape == torch.Size([batch_size, hidden_size])

        # [B, 100, 350]
        local_info = self.local_layer_1(first_rep)
        local_info = self.activation(local_info)  # B'
        local_info = self.local_layer_2(local_info)
        local_info = self.activation(local_info)  # C'
        # print('[B, 100, 350]', local_info.shape)
        assert local_info.shape == torch.Size([batch_size, hidden_size, seq_len])

        assert global_info.shape == torch.Size([batch_size, hidden_size])
        global_info_3d = global_info.view(batch_size, hidden_size, 1).repeat(1, 1, seq_len)
        assert global_info_3d.shape == torch.Size([batch_size, hidden_size, seq_len])

        # [B, 200, T]
        final_rep = torch.cat((global_info_3d, local_info), dim=1)  # D
        assert final_rep.shape == torch.Size([batch_size, hidden_size * 2, seq_len])

        final_rep = self.dropout(final_rep)

        # [B, 100, T]
        final_rep = self.final_layer_1(final_rep)
        final_rep = self.activation(final_rep)  # E
        assert final_rep.shape == torch.Size([batch_size, hidden_size, seq_len])

        # [B, 1, T]
        final_rep = self.final_layer_2(final_rep)  # F
        assert final_rep.shape == torch.Size([batch_size, 1, seq_len])

        return final_rep


class IMLETopK(torch.autograd.Function):
    k: int = 10
    tau: float = 10.0
    lambda_: float = 1000.0

    @staticmethod
    def sample_gumbel_k(shape: torch.Size,
                        k: int,
                        tau: float,
                        device: torch.device) -> Tensor:
        sog = 0.0
        for t in [i + 1.0 for i in range(0, 10)]:
            concentration = torch.tensor(1.0 / k, dtype=torch.float, device=device)
            rate = torch.tensor(t / k, dtype=torch.float, device=device)

            gamma = Gamma(concentration=concentration, rate=rate)
            sample = gamma.sample(sample_shape=shape).to(device)

            sog = sog + sample
        sog = sog - math.log(10.0)
        sog = tau * (sog / k)
        return sog

    @staticmethod
    def sample(logits: Tensor,
               k: int,
               tau: float,
               samples: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        if samples is None:
            samples = IMLETopK.sample_gumbel_k(logits.shape, k, tau, device=logits.device)
        samples = samples.to(logits.device)

        gumbel_softmax_sample = logits + samples
        scores, _ = torch.topk(gumbel_softmax_sample, k, sorted=True)

        thr_2d = scores[:, -1].view(-1, 1)
        z = (gumbel_softmax_sample >= thr_2d).float()
        return z, samples

    @staticmethod
    def forward(ctx,
                logits: Tensor):
        z, sample = IMLETopK.sample(logits, IMLETopK.k, IMLETopK.tau)
        ctx.save_for_backward(logits, z, sample)
        return z

    @staticmethod
    def backward(ctx,
                 dy: Tensor):
        logits, z, sample = ctx.saved_tensors
        target_logits = logits - (IMLETopK.lambda_ * dy)
        map_dy, _ = IMLETopK.sample(target_logits, IMLETopK.k, IMLETopK.tau, sample)
        grad = z - map_dy
        return grad


class PredictionModel(torch.nn.Module):
    def __init__(self,
                 embedding_weights: Tensor,
                 hidden_dims: int,
                 select_k: int):
        super().__init__()
        self.nb_words = embedding_weights.shape[0]
        self.embedding_dim = embedding_weights.shape[1]
        self.hidden_dims = hidden_dims
        self.select_k = float(select_k)

        self.embeddings = nn.Embedding.from_pretrained(embedding_weights, freeze=True)

        self.layer_1 = nn.Linear(in_features=self.embedding_dim, out_features=self.hidden_dims, bias=True)
        init(self.layer_1)

        self.layer_2 = nn.Linear(in_features=self.hidden_dims, out_features=1, bias=True)
        init(self.layer_2)

        self.activation = nn.ReLU()
        self.output_activation = nn.Sigmoid()

    def forward(self,
                x: Tensor,
                mask: Tensor) -> Tensor:
        # [B, T] -> [B, T, E]
        x_emb = self.embeddings(x)

        # [B, S, E]
        res = x_emb * mask
        # [B, E]
        # res = torch.mean(res, dim=1)
        res = torch.sum(res, dim=1) / self.select_k
        # [B, H]
        res = self.layer_1(res)
        res = self.activation(res)
        # [B, 1]
        res = self.layer_2(res)
        res = self.output_activation(res)
        return res


class Model(torch.nn.Module):
    def __init__(self,
                 embedding_weights: Tensor,
                 hidden_dims: int,
                 kernel_size: int,
                 select_k: int,
                 differentiable_select_k: Optional[Callable[[Tensor], Tensor]] = None):
        super().__init__()
        self.gumbel_selector = GumbelSelector(embedding_weights=embedding_weights, kernel_size=kernel_size)
        self.prediction_model = PredictionModel(embedding_weights=embedding_weights, hidden_dims=hidden_dims, select_k=select_k)
        self.differentiable_select_k = differentiable_select_k

    def z(self, x: Tensor) -> Tensor:
        # [B, 1, T]
        token_logits = self.gumbel_selector(x)
        token_logits = token_logits.to(x.device)
        # [B, T, 1]
        token_logits = token_logits.transpose(1, 2)
        batch_size_ = token_logits.shape[0]
        seq_len_ = token_logits.shape[1]
        assert token_logits.shape[2] == 1
        token_logits = token_logits.view(batch_size_, seq_len_)
        # [B, T]
        if self.differentiable_select_k is None:
            assert False, "This should never happen"
            token_selections = IMLETopK.apply(token_logits)
        else:
            token_selections = self.differentiable_select_k(token_logits)
        return token_selections

    def forward(self,
                x: Tensor) -> Tensor:
        # [B, T]
        token_selections = self.z(x)
        # [B, T, 1]
        token_selections = torch.unsqueeze(token_selections, dim=-1)

        # Now, note that, while x is [B, T], token_selections is [B * S, T, 1],
        # where S is the number of samples drawn by I-MLE during the forward pass.
        # We may need to replicate x S times.

        batch_size = x.shape[0]
        seq_len = x.shape[1]

        assert token_selections.shape[1] == seq_len

        if token_selections.shape[0] > batch_size:
            nb_samples = token_selections.shape[0] // batch_size
            x = x.view(batch_size, 1, seq_len)
            x = x.repeat(1, nb_samples, 1)
            x = x.view(batch_size * nb_samples, seq_len)

        p = self.prediction_model(x=x, mask=token_selections)
        return p


class ConcreteDistribution(nn.Module):
    def __init__(self,
                 tau: float,
                 k: int,
                 eps: float = torch.finfo(torch.float).tiny,
                 device: Optional[torch.device] = None):
        super().__init__()
        self.tau = tau
        self.k = k
        self.eps = eps

        u_min = torch.tensor(self.eps, dtype=torch.float, device=device)
        u_max = torch.tensor(1.0, dtype=torch.float, device=device)
        self.base_distribution = Uniform(u_min, u_max)

    def forward(self,
                logits: Tensor):
        # logits is [B, D]
        logits_shape = logits.shape
        if self.training:
            # Uniform, [B, D]
            u_sample = self.base_distribution.sample(sample_shape=logits_shape).to(logits.device)
            # Gumbel, [B, D]
            g_sample = torch.log(- torch.log(u_sample))
            # [B, D]
            noisy_logits = (g_sample + logits) / self.tau
            # [B, D]
            samples = torch.softmax(noisy_logits, dim=1)
            res = samples
        else:
            # [B, k]
            scores, _ = torch.topk(logits, self.k, sorted=True)
            # [B, 1]
            thr_2d = scores[:, -1].view(-1, 1)
            # [B, D]
            z = (logits >= thr_2d).float()
            res = z
        return res


class SampleSubset(nn.Module):
    def __init__(self,
                 tau: float,
                 k: int,
                 eps: float = torch.finfo(torch.float).tiny,
                 device: Optional[torch.device] = None):
        super().__init__()
        self.tau = tau
        self.k = k
        self.eps = eps

        u_min = torch.tensor(self.eps, dtype=torch.float, device=device)
        u_max = torch.tensor(1.0, dtype=torch.float, device=device)
        self.base_distribution = Uniform(u_min, u_max)

    def gumbel_keys(self, w: Tensor) -> Tensor:
        u_sample = self.base_distribution.sample(sample_shape=w.shape).to(w.device)
        z = torch.log(- torch.log(u_sample))
        return w + z

    def continuous_topk(self,
                        w: Tensor,
                        k: int,
                        t: float) -> Tensor:
        # [B, D]
        one_hot_approximation = torch.zeros_like(w, dtype=torch.float, device=w.device)
        res_lst = []
        for i in range(k):
            # [B, D]
            k_hot_mask = torch.clip(1.0 - one_hot_approximation, min=self.eps)
            w_ = w + torch.log(k_hot_mask)
            # [B, D]
            one_hot_approximation = torch.softmax(w_ / t, dim=-1)
            res_lst += [one_hot_approximation]
        # [B, D]
        return sum(res_lst)

    def forward(self,
                logits: Tensor):
        # logits is [B, D]
        if self.training:
            # [B, D]
            w = self.gumbel_keys(logits)
            # [B, D]
            samples = self.continuous_topk(w, self.k, self.tau)
            res = samples
        else:
            # [B, D]
            scores, _ = torch.topk(logits, self.k, sorted=True)
            thr_2d = scores[:, -1].view(-1, 1)
            discrete_logits = (logits >= thr_2d).float()
            res = discrete_logits
        return res
