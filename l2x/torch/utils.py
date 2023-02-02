# -*- coding: utf-8 -*-

import json

import numpy as np
import random

import torch
from torch import nn, Tensor
from torch.distributions.gamma import Gamma

from torch.distributions import Uniform

import math

from l2x.utils import pad_sequences

from typing import Optional, Tuple, Callable

import logging

logger = logging.getLogger(__name__)


def set_seed(seed: int, is_deterministic: bool = True):
    # set the seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if is_deterministic is True:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    return


def subset_precision(model, aspect, id_to_word, word_to_id, select_k, device: torch.device, max_len: int = 350):
    data = []
    num_annotated_reviews = 0
    with open("data/annotations.json") as fin:
        for line in fin:
            item = json.loads(line)
            data.append(item)
            num_annotated_reviews = num_annotated_reviews + 1

    selected_word_counter = 0
    correct_selected_counter = 0

    for anotr in range(num_annotated_reviews):
        ranges = data[anotr][str(aspect)]  # the aspect id
        text_list = data[anotr]['x']
        review_length = len(text_list)

        list_test = []
        tokenid_list = [word_to_id.get(token, 0) for token in text_list]
        list_test.append(tokenid_list)

        # X_test_subset = np.asarray(list_test)
        # X_test_subset = sequence.pad_sequences(X_test_subset, maxlen=350)

        X_test_subset = pad_sequences(list_test, max_len=max_len)
        X_test_subset_t = torch.tensor(X_test_subset, dtype=torch.long, device=device)

        with torch.inference_mode():
            model.eval()
            prediction = model.z(X_test_subset_t)

        x_val_selected = prediction[0].cpu().numpy() * X_test_subset

        # [L,]
        selected_words = np.vectorize(id_to_word.get)(x_val_selected)[0][-review_length:]
        selected_nonpadding_word_counter = 0

        for i, w in enumerate(selected_words):
            if w != '<PAD>':  # we are nice to the L2X approach by only considering selected non-pad tokens
                selected_nonpadding_word_counter = selected_nonpadding_word_counter + 1
                for r in ranges:
                    rl = list(r)
                    if i in range(rl[0], rl[1]):
                        correct_selected_counter = correct_selected_counter + 1
        # we make sure that we select at least 10 non-padding words
        # if we have more than select_k non-padding words selected, we allow it but count that in
        selected_word_counter = selected_word_counter + max(selected_nonpadding_word_counter, select_k)

    return correct_selected_counter / selected_word_counter
