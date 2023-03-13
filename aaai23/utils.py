# -*- coding: utf-8 -*-

import numpy as np

from typing import List, Optional

import logging

logger = logging.getLogger(__name__)


def pad_sequences(sequences: List[List[int]],
                  max_len: Optional[int] = None,
                  padding: str = 'pre') -> np.ndarray:
    if max_len is None:
        max_len = max([len(seq) for seq in sequences])
    nb_sequences = len(sequences)
    res = np.zeros(shape=(nb_sequences, max_len), dtype=int)
    for i, seq in enumerate(sequences):
        seq_len = len(seq)
        if padding in {'pre'}:
            res[i, max_len - min(seq_len, max_len):max_len] = seq[0:max_len]
        else:
            res[i, 0:min(seq_len, max_len)] = seq[0:max_len]
    return res
