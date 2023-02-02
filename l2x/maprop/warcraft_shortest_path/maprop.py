# -*- coding: utf-8 -*-

import numpy as np
import torch

from torch import Tensor

from l2x.maprop.blackbox.losses import HammingLoss

from l2x.maprop.warcraft_shortest_path.trainers import ShortestPathAbstractTrainer

from l2x.maprop.blackbox.dijkstra import get_solver
from l2x.maprop.utils import maybe_parallelize

from l2x.maprop.models import get_model

from imle.aimle import aimle
from imle.ste import ste

from imle.target import TargetDistribution, AdaptiveTargetDistribution
from imle.noise import GumbelNoiseDistribution, SumOfGammaNoiseDistribution

from typing import Dict, Any, Callable


def translate_weights(weights: np.ndarray) -> np.ndarray:
    # Weights can be negative - shift them so they are positive
    weights_shp = weights.shape
    batch_size = weights_shp[0]
    weights_2d = weights.reshape(batch_size, -1)
    instance_min = np.amin(weights_2d, axis=-1)
    res_2d = weights_2d.T - np.minimum(instance_min, 0).T
    return res_2d.T.reshape(weights_shp)


def sanitise_weights(weights: np.ndarray) -> np.ndarray:
    # Weights can be negative - in that case, clip them to 0
    res = np.maximum(weights, 0)
    return res


def map_estimator(weights: Tensor,
                  neighbourhood_fn: str = "8-grid") -> Tensor:
    weights_np = weights.detach().cpu().numpy()
    # weights_np = translate_weights(weights_np)
    weights_np = sanitise_weights(weights_np)

    solver = get_solver(neighbourhood_fn)
    suggested_tours = np.asarray(maybe_parallelize(solver, arg_list=list(weights_np)))

    res = torch.from_numpy(suggested_tours).float().to(weights.device)
    return res


class DijkstraMAP(ShortestPathAbstractTrainer):
    def __init__(self, *, l1_regconst, lambda_val,
                 mode,
                 **kwargs):
        super().__init__(**kwargs)
        self.l1_regconst = l1_regconst
        self.lambda_val = lambda_val

        self.mode = mode
        self.target_distribution = None
        self.noise_distribution = None

        print(f'MAP-BACKPROP MODE: {self.mode}')

        def bb_dijkstra(_weights: Tensor) -> Tensor:
            _solver = get_solver(self.neighbourhood_fn)
            _weights_np = _weights.detach().cpu().numpy()
            _paths = np.asarray(maybe_parallelize(_solver, arg_list=list(_weights_np)))
            _res = torch.from_numpy(_paths).float().to(_weights.device)
            return _res

        self.bb_dijkstra = bb_dijkstra
        self.loss_fn = HammingLoss()

        if 'objective_type' in self.mode and self.mode.objective_type in {'cost', 'cost2'}:
            print(f'OBJECTIVE TYPE: {self.mode.objective_type}')

        print("META:", self.metadata)

    def build_model(self, model_name, arch_params):
        self.model = get_model(
            model_name, out_features=self.metadata["output_features"], in_channels=self.metadata["num_channels"], arch_params=arch_params
        )

    def forward_pass(self, input, true_shortest_paths, train, i, true_weights=None):
        output = self.model(input)
        # make grid weights positive
        output = torch.abs(output)
        weights = output.reshape(-1, output.shape[-1], output.shape[-1])

        assert len(weights.shape) == 3, f"{str(weights.shape)}"

        is_training = self.model.training

        if is_training:

            assert 'target' in self.mode
            if self.target_distribution is None:
                if self.mode.target in {'standard'}:
                    self.target_distribution = TargetDistribution(alpha=1.0, beta=self.lambda_val, do_gradient_scaling=True)
                elif self.mode.target in {'adaptive'}:
                    assert 'lambda_update_step' in self.mode
                    lambda_update_step = self.mode.lambda_update_step
                    self.target_distribution = AdaptiveTargetDistribution(initial_beta=self.lambda_val,
                                                                          beta_update_step=lambda_update_step)
                else:
                    assert False, 'Missing target distribution'

            assert 'noise_distribution' in self.mode
            if self.noise_distribution is None:
                if self.mode.noise_distribution in {'gumbel'}:
                    self.noise_distribution = GumbelNoiseDistribution()
                elif self.mode.noise_distribution in {'sog'}:
                    height_size = weights.shape[1]
                    k_ = int(height_size * 1.3)
                    self.noise_distribution = SumOfGammaNoiseDistribution(k=k_)
                elif self.mode.noise_distribution in {'none'}:
                    self.noise_distribution = None
                else:
                    assert False, 'Missing noise distribution'

            assert 'noise_temperature' in self.mode
            noise_temperature = self.mode.noise_temperature

            assert 'is_symmetric' in self.mode
            is_symmetric = self.mode.is_symmetric

            assert 'nb_samples' in self.mode
            nb_samples = self.mode.nb_samples

            assert 'method' in self.mode
            if self.mode.method in {'imle'}:
                @aimle(target_distribution=self.target_distribution, noise_distribution=self.noise_distribution,
                       theta_noise_temperature=noise_temperature, target_noise_temperature=noise_temperature,
                       symmetric_perturbation=is_symmetric, nb_samples=nb_samples, _is_minimization=True)
                def diff_function(weights_t: Tensor) -> Tensor:
                    return map_estimator(weights_t, self.neighbourhood_fn)
            elif self.mode.method in {'ste'}:
                @ste(noise_distribution=self.noise_distribution, noise_temperature=noise_temperature,
                     nb_samples=nb_samples)
                def diff_function(weights_t: Tensor) -> Tensor:
                    return map_estimator(weights_t, self.neighbourhood_fn)
            elif self.mode.method in {'none'}:
                def diff_function(weights_t: Tensor) -> Tensor:
                    return weights_t
            else:
                assert False, f'Unknown method {self.mode.method}'

            shortest_paths = diff_function(weights)
        else:
            shortest_paths = map_estimator(weights, self.neighbourhood_fn)

        loss = self.loss_fn(shortest_paths, true_shortest_paths)

        logger = self.train_logger if train else self.val_logger

        last_suggestion = {
            "suggested_weights": weights,
            "suggested_path": shortest_paths
        }

        accuracy = (torch.abs(shortest_paths - true_shortest_paths) < 0.5).to(torch.float32).mean()
        extra_loss = self.l1_regconst * torch.mean(output)
        loss += extra_loss

        return loss, accuracy, last_suggestion
