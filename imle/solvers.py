# -*- coding: utf-8 -*-

import torch
from torch import Tensor

import logging

logger = logging.getLogger(__name__)


def select_k(logits: Tensor, k: int) -> Tensor:
    scores, indices = torch.topk(logits, k, sorted=True)
    mask = torch.zeros_like(logits, device=logits.device).scatter_(-1, indices, 1.0)
    return mask


def mathias_select_k(logits: Tensor, k: int) -> Tensor:
    scores, indices = torch.topk(logits, k, sorted=True)
    thr_2d = scores[:, -1].view(-1, 1)
    return (logits >= thr_2d).float()
