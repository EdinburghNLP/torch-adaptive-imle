#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

import numpy as np

import subprocess
import pytest


def test_cli_nri_v1():
    env = os.environ.copy()
    env['PYTHONPATH'] = '.'
    env['WANDB_MODE'] = 'offline'

    cmd_str = 'python3 ./cli/nri-cli.py --suffix _novar_1skip_10t_1r_graph10 --timesteps 10 ' \
              '--prediction_steps 9 --sst tree --symmeterize_logits True --lr 0.0001 --temp 0.5 ' \
              '--eps_for_finitediff 1.0 --cuda False --experiments_folder ' \
              '../exp_nri/nri_T=10_eps=1.0_lr=0.0001_sst=indep_temp=0.5 --use_cpp_for_sampling True ' \
              '--eval_every -1 --batch_size 10 --num_iterations 32 --verbose True --max-steps 32'

    cmd = cmd_str.split()

    p = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()

    print(out)
    print(err)

    sys.stdout = sys.stderr

    lines = out.decode("utf-8").split("\n")

    f1 = f2 = f3 = f4 = False

    for line in lines:
        if 'Epoch 0\tBatch 0\t' in line:
            value = float(line.split()[5])
            np.testing.assert_allclose(value, 25763.42578, atol=1e-3, rtol=1e-3)
            f1 = True
        if 'Epoch 0\tBatch 10\t' in line:
            value = float(line.split()[5])
            np.testing.assert_allclose(value, 17898.48828, atol=1e-3, rtol=1e-3)
            f2 = True
        if 'Epoch 0\tBatch 20\t' in line:
            value = float(line.split()[5])
            np.testing.assert_allclose(value, 11943.35840, atol=1e-3, rtol=1e-3)
            f3 = True
        if 'Epoch 0\tBatch 30\t' in line:
            value = float(line.split()[5])
            np.testing.assert_allclose(value, 10786.67383, atol=1e-3, rtol=1e-3)
            f4 = True

    assert f1 is True
    assert f2 is True
    assert f3 is True
    assert f4 is True


if __name__ == '__main__':
    pytest.main([__file__])
    # test_cli_v1()
