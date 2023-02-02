#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

import time
import numpy as np

import argparse

import torch

from torch import optim, Tensor
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

from imle.imle import imle
from imle.aimle import aimle
from imle.ste import ste
from imle.target import TargetDistribution, AdaptiveTargetDistribution
from imle.noise import BaseNoiseDistribution, SumOfGammaNoiseDistribution, GumbelNoiseDistribution
from imle.solvers import mathias_select_k

from sklearn.model_selection import train_test_split

from l2x.torch.utils import set_seed, subset_precision
from l2x.torch.modules import Model, ConcreteDistribution, SampleSubset, IMLETopK
from l2x.utils import pad_sequences

from typing import Optional, Callable

import socket
import wandb

import logging

logger = logging.getLogger(os.path.basename(sys.argv[0]))


class DifferentiableSelectKModel(torch.nn.Module):
    def __init__(self,
                 diff_fun: Callable[[Tensor], Tensor],
                 fun: Callable[[Tensor], Tensor]):
        super().__init__()
        self.diff_fun = diff_fun
        self.fun = fun

    def forward(self, logits: Tensor) -> Tensor:
        return self.diff_fun(logits) if self.training else self.fun(logits)


def evaluate(model_eval: Model,
             x_eval: np.ndarray,
             y_eval: np.ndarray,
             device: torch.device) -> float:
    mse = torch.nn.MSELoss()
    x_eval_t = torch.tensor(x_eval, dtype=torch.long, device=device)
    y_eval_t = torch.tensor(y_eval, dtype=torch.float, device=device)
    eval_dataset = TensorDataset(x_eval_t, y_eval_t)
    eval_loader = DataLoader(eval_dataset, batch_size=100, shuffle=False)
    with torch.inference_mode():
        model_eval.eval()
        p_eval_lst = []
        for X, y in eval_loader:
            p_eval_lst += model_eval(x=X).view(-1).tolist()
        p_eval_t = torch.tensor(p_eval_lst, dtype=torch.float, requires_grad=False, device=device)
        mse_value = mse(p_eval_t, y_eval_t)
    return mse_value.item()


def main(argv):
    parser = argparse.ArgumentParser('PyTorch I-MLE/BeerAdvocate', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--aspect', '-a', action='store', type=int, default=1, help='Aspect')
    parser.add_argument('--epochs', '-e', action='store', type=int, default=20, help='Epochs')
    parser.add_argument('--batch-size', '-b', action='store', type=int, default=40, help='Batch Size')
    parser.add_argument('--kernel-size', '-k', action='store', type=int, default=3, help='Kernel Size')
    parser.add_argument('--hidden-dims', '-H', action='store', type=int, default=250, help='Hidden Dimensions')
    parser.add_argument('--max-len', '-m', action='store', type=int, default=350, help='Maximum Sequence Length')
    parser.add_argument('--select-k', '-K', action='store', type=int, default=10, help='Select K')

    parser.add_argument("--checkpoint", "-c", action='store', type=str, default='models/model.pt')
    parser.add_argument("--reruns", "-r", action='store', type=int, default=10)
    parser.add_argument("--method", "-M", type=str, choices=['sst', 'imle', 'imletopk', 'aimle', 'ste', 'softsub'],
                        default='imle', help="Method (SST, IMLE, AIMLE, STE, SoftSub)")

    parser.add_argument('--aimle-symmetric', action='store_true', default=False)
    parser.add_argument('--aimle-target', type=str, choices=['standard', 'adaptive'], default='standard')

    parser.add_argument('--imle-noise', type=str, choices=['none', 'sog', 'gumbel'], default='sog')
    parser.add_argument('--imle-samples', action='store', type=int, default=1)
    parser.add_argument('--imle-input-temperature', action='store', type=float, default=0.0)
    parser.add_argument('--imle-output-temperature', action='store', type=float, default=10.0)
    parser.add_argument('--imle-lambda', action='store', type=float, default=1000.0)

    parser.add_argument('--aimle-beta-update-step', action='store', type=float, default=0.0001)
    parser.add_argument('--aimle-beta-update-momentum', action='store', type=float, default=0.0)
    parser.add_argument('--aimle-target-norm', action='store', type=float, default=1.0)

    parser.add_argument('--sst-temperature', action='store', type=float, default=0.1)

    parser.add_argument('--softsub-temperature', action='store', type=float, default=0.5)

    parser.add_argument('--ste-noise', type=str, choices=['none', 'sog', 'gumbel'], default='sog')
    parser.add_argument('--ste-temperature', action='store', type=float, default=0.0)

    parser.add_argument('--debug', '-D', action='store_true', default=False)
    parser.add_argument('--max-iterations', action='store', type=int, default=None)
    parser.add_argument('--gradient-scaling', action='store_true', default=False)

    args = parser.parse_args(argv)

    if args.debug is True:
        torch.autograd.set_detect_anomaly(True)

    hostname = socket.gethostname()
    print(f'Hostname: {hostname}')

    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')

    aspect = args.aspect

    input_path_train = "data/reviews.aspect" + str(aspect) + ".train.txt"
    input_path_validation = "data/reviews.aspect" + str(aspect) + ".heldout.txt"

    # the dictionary mapping words to their IDs
    word_to_id = dict()
    token_id_counter = 3

    with open(input_path_train) as fin:
        for line in fin:
            y, sep, text = line.partition("\t")
            token_list = text.split(" ")
            for token in token_list:
                if token not in word_to_id:
                    word_to_id[token] = token_id_counter
                    token_id_counter = token_id_counter + 1

    word_to_id["<PAD>"] = 0
    word_to_id["<START>"] = 1
    word_to_id["<UNK>"] = 2

    # Set parameters:
    method_name = args.method

    # max_features = token_id_counter + 1
    maxlen = args.max_len
    batch_size = args.batch_size
    embedding_dims = 200
    kernel_size = args.kernel_size
    hidden_dims = args.hidden_dims
    epochs = args.epochs
    select_k = args.select_k  # Number of selected words by the methods
    checkpoint_path = args.checkpoint

    id_to_word = {value: key for key, value in word_to_id.items()}

    X_train_list = []
    Y_train_list = []
    # now we iterate again to assign IDs
    with open(input_path_train) as fin:
        for line in fin:
            y, sep, text = line.partition("\t")
            token_list = text.split(" ")
            tokenid_list = [word_to_id[token] for token in token_list]
            X_train_list.append(tokenid_list)

            # extract the normalized [0,1] value for the aspect
            y = [float(v) for v in y.split()]
            Y_train_list.append(y[aspect])

    X_train = pad_sequences(X_train_list, max_len=maxlen)
    Y_train = np.asarray(Y_train_list)

    X_train_t = torch.tensor(X_train, dtype=torch.long, device=device)
    Y_train_t = torch.tensor(Y_train, dtype=torch.float, device=device)
    train_dataset = TensorDataset(X_train_t, Y_train_t)

    print("Loading heldout data...")
    X_val_list = []
    Y_val_list = []

    # now we iterate again to assign IDs
    with open(input_path_validation) as fin:
        for line in fin:
            y, sep, text = line.partition("\t")
            token_list = text.split(" ")
            tokenid_list = [word_to_id.get(token, 2) for token in token_list]
            X_val_list.append(tokenid_list)

            # extract the normalized [0,1] value for the aspect
            y = [float(v) for v in y.split()]
            Y_val_list.append(y[aspect])

    X_val_both = pad_sequences(X_val_list, max_len=maxlen)
    Y_val_both = np.asarray(Y_val_list)

    # this cell loads the word embeddings from the external data
    embeddings_index = {}
    with open("data/review+wiki.filtered.200.txt") as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    print('Found %s word vectors.' % len(embeddings_index))

    embedding_matrix = np.zeros((len(word_to_id) + 1, embedding_dims))
    for word, i in word_to_id.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    print('Creating model...')

    val_mse_lst = []
    test_mse_lst = []
    subset_precision_lst = []

    embedding_matrix_t = torch.tensor(embedding_matrix, dtype=torch.float, requires_grad=False, device=device)

    loss_function = torch.nn.MSELoss()
    loss_function_nored = torch.nn.MSELoss(reduction='none')

    # here we can now iterate a few times to compute statistics
    for seed in range(args.reruns):
        wandb.init(project="beeradv-l2x", name=f'{method_name}-{seed}')

        wandb.config.update(args)
        wandb.config.update({'hostname': hostname, 'seed': seed})

        set_seed(seed, is_deterministic=True)

        # create a new validation/test split
        X_val, X_test, Y_val, Y_test = train_test_split(X_val_both, Y_val_both, test_size=0.5, random_state=seed)

        print('Initialising the model ..')

        def name_to_distribution(distribution_name: str) -> Optional[BaseNoiseDistribution]:
            if distribution_name in {'none'}:
                noise_distribution = None
            elif distribution_name in {'sog'}:
                noise_distribution = SumOfGammaNoiseDistribution(k=select_k, nb_iterations=10, device=device)
            elif distribution_name in {'gumbel'}:
                noise_distribution = GumbelNoiseDistribution(device=device)
            else:
                assert False, f'Noise model not supported: {distribution_name}'
            return noise_distribution

        blackbox_function = lambda logits: mathias_select_k(logits, k=select_k)

        if method_name in {'imle'}:
            nb_samples = args.imle_samples
            imle_input_temp = args.imle_input_temperature
            imle_output_temp = args.imle_output_temperature
            imle_lambda = args.imle_lambda

            target_distribution = TargetDistribution(alpha=1.0, beta=imle_lambda, do_gradient_scaling=args.gradient_scaling)
            noise_distribution = name_to_distribution(args.imle_noise)

            @imle(target_distribution=target_distribution, noise_distribution=noise_distribution, nb_samples=nb_samples,
                  theta_noise_temperature=imle_input_temp, target_noise_temperature=imle_output_temp)
            def imle_select_k(logits: Tensor) -> Tensor:
                return mathias_select_k(logits, k=select_k)

            differentiable_select_k = DifferentiableSelectKModel(imle_select_k, blackbox_function)

        elif method_name in {'imletopk'}:
            IMLETopK.k = select_k
            IMLETopK.tau = args.imle_output_temperature
            IMLETopK.lambda_ = args.imle_lambda

            differentiable_select_k = DifferentiableSelectKModel(IMLETopK.apply, blackbox_function)

        elif method_name in {'aimle'}:
            nb_samples = args.imle_samples
            imle_input_temp = args.imle_input_temperature
            imle_output_temp = args.imle_output_temperature
            imle_lambda = args.imle_lambda

            if args.aimle_target in {'standard'}:
                target_distribution = TargetDistribution(alpha=1.0,
                                                         beta=imle_lambda,
                                                         do_gradient_scaling=args.gradient_scaling)
            elif args.aimle_target in {'adaptive'}:
                target_distribution = AdaptiveTargetDistribution(initial_alpha=1.0,
                                                                 initial_beta=imle_lambda,
                                                                 beta_update_step=args.aimle_beta_update_step,
                                                                 beta_update_momentum=args.aimle_beta_update_momentum,
                                                                 target_norm=args.aimle_target_norm)
            else:
                assert False, f'Do not know how to handle {args.aimle_target} as target distribution'
            noise_distribution = name_to_distribution(args.imle_noise)

            @aimle(target_distribution=target_distribution, noise_distribution=noise_distribution,
                   nb_samples=nb_samples,
                   theta_noise_temperature=imle_input_temp, target_noise_temperature=imle_output_temp,
                   symmetric_perturbation=args.aimle_symmetric)
            def aimle_select_k(logits: Tensor) -> Tensor:
                return mathias_select_k(logits, k=select_k)

            differentiable_select_k = DifferentiableSelectKModel(aimle_select_k, blackbox_function)

        elif method_name in {'ste'}:
            noise_distribution = name_to_distribution(args.ste_noise)

            @ste(noise_distribution=noise_distribution, noise_temperature=args.ste_temperature)
            def ste_select_k(logits: Tensor) -> Tensor:
                return mathias_select_k(logits, k=select_k)

            differentiable_select_k = DifferentiableSelectKModel(ste_select_k, blackbox_function)

        elif method_name in {'sst'}:
            tau = args.sst_temperature

            differentiable_select_k = ConcreteDistribution(tau=tau, k=select_k, device=device)
        elif method_name in {'softsub'}:
            tau = args.softsub_temperature

            differentiable_select_k = SampleSubset(tau=tau, k=select_k, device=device)
        else:
            assert False, f'Method not supported: {method_name}'

        model = Model(embedding_weights=embedding_matrix_t,
                      hidden_dims=hidden_dims,
                      kernel_size=kernel_size,
                      select_k=select_k,
                      differentiable_select_k=differentiable_select_k).to(device)

        print('Model:')
        group_name_to_nparams = dict()
        for name in model.state_dict():
            group_name = name.split('.')[0].strip()
            pt = model.state_dict()[name]
            print(f'\t{name}\t{pt.size()}\t{pt.numel()}')
            group_name_to_nparams[group_name] = group_name_to_nparams.get(group_name, 0) + pt.numel()

        print('Model modules:')
        for name, nparams in group_name_to_nparams.items():
            print(f'\t{name}\t{nparams}')

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        optimizer = optim.Adam(model.parameters(), lr=0.001, eps=1e-7)

        st = time.time()
        best_val_mse = None

        for epoch_no in range(1, epochs + 1):
            epoch_loss_values = []

            for i, (X, y) in enumerate(train_loader):
                # Used for unit tests
                if args.max_iterations is not None and i > args.max_iterations:
                    break

                model.train()
                p = model(x=X).view(-1)

                # Now, note that, while y is [B], p is [B * S], where S is the number of samples
                # drawn by I-MLE during the forward pass. We may need to replicate y S times.
                nb_samples = p.shape[0] // y.shape[0]
                if p.shape[0] > y.shape[0]:
                    assert method_name in {'imle', 'aimle'} and args.imle_samples > 1, "p.shape and y.shape differ"
                    y = y.view(batch_size, 1)
                    y = y.repeat(1, nb_samples)
                    y = y.view(batch_size * nb_samples)

                # XXX About the loss values, remember we should sum over S and aggregate over B
                assert nb_samples > 0
                if nb_samples == 1:
                    loss = loss_function(p, y)
                else:
                    loss = loss_function_nored(p, y)
                    loss = loss.view(-1, nb_samples).sum(axis=1).mean(axis=0)

                loss_value = loss.item()

                if args.debug is True:
                    logger.info(f'Epoch {epoch_no}/{epochs}\tIteration {i + 1}\tLoss value: {loss_value:.4f}')

                epoch_loss_values += [loss_value]

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            loss_mean, loss_std = np.mean(epoch_loss_values), np.std(epoch_loss_values)
            logger.info(f'Epoch {epoch_no}/{epochs}\tLoss {loss_mean:.4f} ± {loss_std:.4f}')

            # Checkpointing
            val_mse = evaluate(model, X_val, Y_val, device=device)
            test_mse = evaluate(model, X_test, Y_test, device=device)

            if best_val_mse is None or val_mse <= best_val_mse:
                print(f'Saving new checkpoint -- new best validation MSE: {val_mse:.5f}')
                torch.save({'model_state_dict': model.state_dict()}, checkpoint_path)
                best_val_mse = val_mse

            wandb.log({'seed': seed, 'val_mse': val_mse, 'test_mse': test_mse, 'loss_mean': loss_mean}, step=epoch_no)

        duration = time.time() - st
        print(f'[{seed}] Training time is {duration} ms')

        if os.path.isfile(checkpoint_path):
            print(f'Loading checkpoint at {checkpoint_path} ..')
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])

        val_mse = evaluate(model, X_val, Y_val, device=device) * 100.0
        print(f"[{seed}] Validation MSE: {val_mse:.5f}")
        val_mse_lst += [val_mse]

        test_mse = evaluate(model, X_test, Y_test, device=device) * 100.0
        print(f"[{seed}] Test MSE: {test_mse:.5f}")
        test_mse_lst += [test_mse]

        subset_prec = subset_precision(model, aspect, id_to_word, word_to_id, select_k,
                                       device=device, max_len=maxlen) * 100.0
        print(f"[{seed}] Subset precision: {subset_prec:.5f}")
        subset_precision_lst += [subset_prec]

        wandb.log({'best_val_mse': val_mse, 'best_test_mse': test_mse, 'best_subset_prec': subset_prec})

        wandb.finish()

    print(f'Final Subset Precision List: {np.mean(subset_precision_lst):.5f} ± {np.std(subset_precision_lst):.5f}')
    print(f'Final Validation MSE List: {np.mean(val_mse_lst):.5f} ± {np.std(val_mse_lst):.5f}')
    print(f'Final Test MSE List: {np.mean(test_mse_lst):.5f} ± {np.std(test_mse_lst):.5f}')

    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    print('Experiment completed.')


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    main(sys.argv[1:])
