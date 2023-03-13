# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F

from torch import nn, Tensor

from typing import Callable, Tuple

import logging

logger = logging.getLogger(__name__)


def init(layer: nn.Module):
    if isinstance(layer, nn.Conv1d) or isinstance(layer, nn.Linear):
        torch.nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain('relu'))
        torch.nn.init.zeros_(layer.bias)
    else:
        assert f'Do not know how to deal with {type(layer)}'


class Encoder(nn.Module):
    def __init__(self,
                 input_dim: int,
                 code_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.code_dim = code_dim

        self.linear1 = torch.nn.Linear(self.input_dim, 512, bias=True)
        init(self.linear1)
        self.linear2 = torch.nn.Linear(512, 256, bias=True)
        init(self.linear2)
        self.linear3 = torch.nn.Linear(256, self.code_dim, bias=True)
        init(self.linear3)

    def forward(self, x: Tensor) -> Tensor:
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        return self.linear3(x)


class Decoder(nn.Module):
    def __init__(self,
                 input_dim: int,
                 code_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.code_dim = code_dim

        self.linear1 = torch.nn.Linear(self.code_dim, 256, bias=True)
        init(self.linear1)
        self.linear2 = torch.nn.Linear(256, 512, bias=True)
        init(self.linear2)
        self.linear3 = torch.nn.Linear(512, self.input_dim, bias=True)
        init(self.linear3)

    def forward(self, x: Tensor) -> Tensor:
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        return self.linear3(x)


class DiscreteVAE(nn.Module):
    def __init__(self,
                 input_dim: int,
                 n: int = 20,
                 m: int = 20):
        super().__init__()
        self.input_dim = input_dim
        self.n = n
        self.m = m
        self.code_dim = self.m * self.n

        self.encoder = Encoder(input_dim=self.input_dim, code_dim=self.code_dim)
        self.decoder = Decoder(input_dim=self.input_dim, code_dim=self.code_dim)

    def forward(self,
                x: Tensor,
                code_generator: Callable[[Tensor], Tensor]) -> Tuple[Tensor, Tensor]:
        # x is # [B, H * W]
        batch_size = x.shape[0]

        # [B, M * N]
        logits_2d = self.encoder(x)

        assert len(logits_2d.shape) == 2
        assert logits_2d.shape[0] == batch_size
        assert logits_2d.shape[1] == self.m * self.n

        # [B * M, N]
        logits_m_2d = logits_2d.view(batch_size * self.m, self.n)

        # [B * M * S, N]
        code_m_2d = code_generator(logits_m_2d)

        # print(f'Expected: {batch_size * self.m} x {self.n}')
        # print(f'Got: {code_m_2d.shape}')

        # Note: if we are using *IMLE and nb_samples > 1,
        # code_generator may return a [B * S * M, N] tensor,
        # where S = nb_samples

        # [B, M, S, N]
        code_4d = code_m_2d.view(batch_size, self.m, -1, self.n)
        nb_samples = code_4d.shape[2]

        # [B, S, M, N]
        code_4d = torch.transpose(code_4d, 1, 2)

        # print(code_4d[0, 0])

        # [B * S, M * N]
        # code_2d = code_m_2d.view(batch_size * nb_samples, self.m * self.n)

        code_2d = code_4d.reshape(batch_size * nb_samples, self.m * self.n)

        assert len(code_2d.shape) == 2
        assert code_2d.shape[0] == batch_size * nb_samples
        assert code_2d.shape[1] == self.m * self.n

        # [B, H * W]
        reconstruction = self.decoder(code_2d)

        assert x.shape[0] * nb_samples == reconstruction.shape[0]
        assert x.shape[1] == reconstruction.shape[1]

        return logits_2d, reconstruction
