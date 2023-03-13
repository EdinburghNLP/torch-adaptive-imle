#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import itertools
import os
import os.path

import sys
import logging


def cartesian_product(dicts):
    return (dict(zip(dicts, x)) for x in itertools.product(*dicts.values()))


def summary(configuration):
    res = configuration['ckp'].split('/')[-1] + '_' + str(configuration['seed'])
    return res


def to_cmd(c, _path=None):
    command = f'{c["cmd"]} --experiments_folder {c["ckp"]}_{c["seed"]} --seed {c["seed"]}'

    if '--timesteps 10' in command:
        assert 'T10' in c["ckp"]
    elif '--timesteps 20' in command:
        assert 'T20' in c["ckp"]
    else:
        assert False

    assert '--hard True' in command

    if '_sym_' in c["ckp"]:
        assert '--aimle-symmetric' in command
    elif '_nosym_' in c["ckp"]:
        assert '--aimle-symmetric' not in command
    else:
        assert False, 'Symmetric or not?'

    if 'aimle-target adaptive' in command:
        assert '/aimle_' in c["ckp"]

    if 'aimle-target standard' in command:
        assert '/imle_' in c["ckp"] or '/sst_' in c["ckp"]

    if '--method aimle' in command:
        assert 'imle_' in c["ckp"]

    if '--method sst' in command:
        assert 'sst_' in c["ckp"]

    assert ('--method aimle' in command) != ('--method sst' in command)

    return command


def to_logfile(c, path):
    outfile = "{}/nribest_beaker_v3.{}.log".format(path, summary(c).replace("/", "_"))
    return outfile


def main(argv):
    hyp_space_aimle_sym_10 = dict(
        cmd=['PYTHONPATH=. python3 ./cli/torch-nri-cli.py --suffix _novar_1skip_10t_1r_graph10 --timesteps 10 '
             '--prediction_steps 9 --sst tree --relaxation exp_family_entropy --max_range 15 --symmeterize_logits True '
             '--lr 0.0005 --temp 0.1 --eps_for_finitediff 1.0 --cuda True '
             '--experiments_folder ../exp_nri/nri_T=10_bu=0.001_eps=1.0_hard=True_imle_samples=1_lmbda=0.0_lr=0.0005_method=aimle_noise=sog_noise_temp=0.1_scaling=False_sst=tree_symmetric=True_target=adaptive_temp=0.1 '
             '--use_cpp_for_sampling True --method aimle --imle-lambda 0.0 --imle-lambda-update-step 0.001 --imle-noise sog '
             '--imle-noise-temperature 0.1 --aimle-symmetric --aimle-target adaptive --imle-samples 1 --hard True --st True'],
        seed=[0, 1, 2, 3, 4, 5, 6, 7],
        ckp=['../exp_nri/aimle_v3_T10_sym_best']
    )

    hyp_space_aimle_nosym_10 = dict(
        cmd=['PYTHONPATH=. python3 ./cli/torch-nri-cli.py --suffix _novar_1skip_10t_1r_graph10 --timesteps 10 '
             '--prediction_steps 9 --sst tree --relaxation exp_family_entropy --max_range 15 --symmeterize_logits True '
             '--lr 0.0001 --temp 0.1 --eps_for_finitediff 1.0 --cuda True '
             '--experiments_folder ../exp_nri/nri_T=10_bu=0.001_eps=1.0_hard=True_imle_samples=1_lmbda=0.0_lr=0.0001_method=aimle_noise=sog_noise_temp=1.0_scaling=False_sst=tree_symmetric=False_target=adaptive_temp=0.1 '
             '--use_cpp_for_sampling True --method aimle --imle-lambda 0.0 --imle-lambda-update-step 0.001 --imle-noise sog '
             '--imle-noise-temperature 1.0 --aimle-target adaptive --imle-samples 1 --hard True --st True'],
        seed=[0, 1, 2, 3, 4, 5, 6, 7],
        ckp=['../exp_nri/aimle_v3_T10_nosym_best']
    )

    hyp_space_aimle_sym_20 = dict(
        cmd=['PYTHONPATH=. python3 ./cli/torch-nri-cli.py --suffix _novar_1skip_20t_1r_graph10 --timesteps 20 '
             '--prediction_steps 10 --sst tree --relaxation exp_family_entropy --max_range 15 --symmeterize_logits True '
             '--lr 0.0005 --temp 0.1 --eps_for_finitediff 1.0 --cuda True '
             '--experiments_folder ../exp_nri/nri_T=20_bu=0.001_eps=1.0_hard=True_imle_samples=1_lmbda=0.0_lr=0.0005_method=aimle_noise=sog_noise_temp=1.0_scaling=False_sst=tree_symmetric=True_target=adaptive_temp=0.1 '
             '--use_cpp_for_sampling True --method aimle --imle-lambda 0.0 --imle-lambda-update-step 0.001 --imle-noise sog '
             '--imle-noise-temperature 1.0 --aimle-symmetric --aimle-target adaptive --imle-samples 1 --hard True --st True'],
        seed=[0, 1, 2, 3, 4, 5, 6, 7],
        ckp=['../exp_nri/aimle_v3_T20_sym_best']
    )

    hyp_space_aimle_nosym_20 = dict(
        cmd=['PYTHONPATH=. python3 ./cli/torch-nri-cli.py --suffix _novar_1skip_20t_1r_graph10 --timesteps 20 '
             '--prediction_steps 10 --sst tree --relaxation exp_family_entropy --max_range 15 --symmeterize_logits True '
             '--lr 0.0005 --temp 0.1 --eps_for_finitediff 1.0 --cuda True '
             '--experiments_folder ../exp_nri/nri_T=20_bu=0.001_eps=1.0_hard=True_imle_samples=1_lmbda=0.0_lr=0.0005_method=aimle_noise=sog_noise_temp=0.1_scaling=False_sst=tree_symmetric=False_target=adaptive_temp=0.1 '
             '--use_cpp_for_sampling True --method aimle --imle-lambda 0.0 --imle-lambda-update-step 0.001 --imle-noise sog '
             '--imle-noise-temperature 0.1 --aimle-target adaptive --imle-samples 1 --hard True --st True'],
        seed=[0, 1, 2, 3, 4, 5, 6, 7],
        ckp=['../exp_nri/aimle_v3_T20_nosym_best']
    )

    configurations = list(cartesian_product(hyp_space_aimle_sym_10)) + \
                     list(cartesian_product(hyp_space_aimle_nosym_10)) + \
                     list(cartesian_product(hyp_space_aimle_sym_20)) + \
                     list(cartesian_product(hyp_space_aimle_nosym_20))

    path = 'logs/nri/nribest_beaker_v3'
    is_rc = False

    # Check that we are on the UCLCS cluster first
    if os.path.exists('/home/pminervi/'):
        is_rc = True
        # If the folder that will contain logs does not exist, create it
        if not os.path.exists(path):
            os.makedirs(path)

    command_lines = set()
    for cfg in configurations:
        logfile = to_logfile(cfg, path)

        completed = False
        if os.path.isfile(logfile):
            with open(logfile, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                completed = 'Optimization Finished' in content

        if not completed:
            command_line = '{} > {} 2>&1'.format(to_cmd(cfg), logfile)
            command_lines |= {command_line}

    # Sort command lines and remove duplicates
    sorted_command_lines = sorted(command_lines)

    import random
    rng = random.Random(0)
    rng.shuffle(sorted_command_lines)

    nb_jobs = len(sorted_command_lines)

    header = """#!/bin/bash -l

#$ -cwd
#$ -S /bin/bash
#$ -o $HOME/array.out
#$ -e $HOME/array.err
#$ -t 1-{}
#$ -l tmem=12G
#$ -l h_rt=48:00:00
#$ -l gpu=true

conda activate gpu

export LANG="en_US.utf8"
export LANGUAGE="en_US:en"
export CUDA_LAUNCH_BLOCKING=1

cd $HOME/workspace/l2x-aimle

""".format(nb_jobs)

    if is_rc:
        print(header)

    for job_id, command_line in enumerate(sorted_command_lines, 1):
        if is_rc:
            print(f'test $SGE_TASK_ID -eq {job_id} && sleep 30 && {command_line}')
        else:
            print(command_line)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1:])
