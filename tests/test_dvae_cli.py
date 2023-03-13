#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

import numpy as np

import subprocess
import pytest


def test_cli_dvae_v1():
    env = os.environ.copy()
    env['PYTHONPATH'] = '.'
    env['WANDB_MODE'] = 'offline'

    cmd_str = 'python3 ./cli/vae-cli.py --imle-samples 1 -e 1 -M imle --imle-temperature 0.0 --imle-noise none'

    cmd = cmd_str.split()

    p = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()

    print(out)
    print(err)

    sys.stdout = sys.stderr

    lines = out.decode("utf-8").split("\n")

    f1 = False

    for line in lines:
        if 'Epoch 1/1\t' in line and 'Training Loss' in line:
            value = float(line.split()[4])
            np.testing.assert_allclose(value, 207.1361, atol=1e-3, rtol=1e-3)
            f1 = True

    assert f1 is True


if __name__ == '__main__':
    pytest.main([__file__])
