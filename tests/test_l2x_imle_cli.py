#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

import numpy as np

import subprocess
import pytest


def test_cli_imle_v1():
    env = os.environ.copy()
    env['PYTHONPATH'] = '.'
    env['WANDB_MODE'] = 'offline'

    cmd_str = 'python3 ./cli/l2x-cli.py -a 1 -e 1 -b 40 -k 3 -H 250 -m 350 -K 10 -r 1 -M imle --imle-samples 1 ' \
              '--imle-noise sog --imle-input-temperature 10.0 --imle-output-temperature 10.0 --imle-lambda 1000.0 -D ' \
              '--max-iterations 10  -c models/test_imle_v1.pt'

    cmd = cmd_str.split()

    p = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()

    print(out)
    print(err)

    sys.stdout = sys.stderr

    lines = out.decode("utf-8").split("\n")

    f1 = f2 = f3 = f4 = f5 = False

    for line in lines:
        if 'Iteration 1\t' in line:
            value = float(line.split()[6])
            np.testing.assert_allclose(value, 0.0957, atol=1e-3, rtol=1e-3)
            f1 = True
        if 'Iteration 10\t' in line:
            value = float(line.split()[6])
            np.testing.assert_allclose(value, 0.0547, atol=1e-3, rtol=1e-3)
            f2 = True
        if '[0] Validation MSE' in line:
            value = float(line.split()[3])
            np.testing.assert_allclose(value, 5.84024, atol=1e-3, rtol=1e-3)
            f3 = True
        if '[0] Test MSE' in line:
            value = float(line.split()[3])
            np.testing.assert_allclose(value, 5.85031, atol=1e-3, rtol=1e-3)
            f4 = True
        if '[0] Subset precision' in line:
            value = float(line.split()[3])
            np.testing.assert_allclose(value, 12.38306, atol=1e-3, rtol=1e-3)
            f5 = True

    assert f1 is True
    assert f2 is True
    assert f3 is True
    assert f4 is True
    assert f5 is True


def test_cli_ste_v1():
    env = os.environ.copy()
    env['PYTHONPATH'] = '.'
    env['WANDB_MODE'] = 'offline'

    cmd_str = 'python3 ./cli/l2x-cli.py -a 1 -e 1 -b 40 -k 3 -H 250 -m 350 -K 10 -r 1 -M ste --imle-samples 1 ' \
              '--imle-noise sog --imle-input-temperature 10.0 --imle-output-temperature 10.0 --imle-lambda 1000.0 -D ' \
              '--max-iterations 10 --ste-temperature 0.0 -c models/test_ste_v1.pt'

    cmd = cmd_str.split()

    p = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()

    print(out)
    print(err)

    sys.stdout = sys.stderr

    lines = out.decode("utf-8").split("\n")

    f1 = f2 = f3 = f4 = f5 = False

    for line in lines:
        if 'Iteration 1\t' in line:
            value = float(line.split()[6])
            np.testing.assert_allclose(value, 0.0950, atol=1e-3, rtol=1e-3)
            f1 = True
        if 'Iteration 10\t' in line:
            value = float(line.split()[6])
            np.testing.assert_allclose(value, 0.0514, atol=1e-3, rtol=1e-3)
            f2 = True
        if '[0] Validation MSE' in line:
            value = float(line.split()[3])
            np.testing.assert_allclose(value, 5.97485, atol=1e-3, rtol=1e-3)
            f3 = True
        if '[0] Test MSE' in line:
            value = float(line.split()[3])
            np.testing.assert_allclose(value, 5.95007, atol=1e-3, rtol=1e-3)
            f4 = True
        if '[0] Subset precision' in line:
            value = float(line.split()[3])
            np.testing.assert_allclose(value, 26.91882, atol=1e-1, rtol=1e-1)
            f5 = True

    assert f1 is True
    assert f2 is True
    assert f3 is True
    assert f4 is True
    assert f5 is True


if __name__ == '__main__':
    pytest.main([__file__])
    # test_cli_v1()
