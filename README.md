# torch-adaptive-imle

Source code for the paper [Adaptive Perturbation-Based Gradient Estimation for Discrete Latent Variable Models](https://arxiv.org/abs/2209.04862), published in the proceedings of the Thirty-Seventh AAAI Conference on Artificial Intelligence (AAAI 2023) -- check out our [AAAI 2023 poster](http://data.neuralnoise.com/AIMLE_AAAI23_poster.pdf), [slides](http://data.neuralnoise.com/AIMLE_AAAI23_slides.pdf), and [AAAI 2023 presentation](https://youtu.be/94MTwQlXrxg).

This work extends our [Implicit MLE](https://arxiv.org/abs/2106.01798) method for back-propagating though black-box combinatorial solvers such as `top-k` functions, shortest path algorithms, and maximum spanning tree algorithms -- for a friendly introduction to Implicit MLE, check [our video](https://www.youtube.com/watch?v=hb2b0K2PTxI) or [Yannic Kilcher's video](https://www.youtube.com/watch?v=W2UT8NjUqrk) on this topic.

Here the `imle/` folder contains a plug-and-play library with decorators for turning arbitrary black-box combinatorial solvers into differentiable neural network layers, similar in spirit to [torch-imle](https://github.com/uclnlp/torch-imle), while `aaai23/` contains the code we used in the experiments of our AAAI 2023 paper.

## Using the code as a library

This code extends the popular [torch-imle](https://github.com/uclnlp/torch-imle) library to adaptively select the optimal target function for the problem at hand.

Sample usage:

```python
import torch
from torch import Tensor
import numpy as np
from imle.aimle import aimle
from imle.target import AdaptiveTargetDistribution

# The initial perturbation size is set to 0.0, and automatically tuned by the model during training
target_distribution = AdaptiveTargetDistribution(initial_alpha=1.0, initial_beta=0.0)

# This function invokes, for example, a shortest path algorithm on the inputs `weights_batch`
def batched_solver(weights_batch: Tensor) -> Tensor:
    weights_batch = weights_batch.detach().cpu().numpy()
    # Call the combinatorial solver -- for example, a shortest path algorithm -- on the input data
    y_batch = np.asarray([solver(w) for w in list(weights_batch)])
    return torch.tensor(y_batch, requires_grad=False)

# Transform the combinatorial solver in a differentiable neural network layer by adding a simple decorator
@aimle(target_distribution=target_distribution)
def differentiable_solver(weights_batch: Tensor) -> Tensor:
    return batched_solver(weights_batch)
```

## Learning to Explain

Downloading the data:

```bash
$ ./get_data.sh
```

Running the experiments:

```bash
$ WANDB_MODE=offline PYTHONPATH=. python3 ./cli/l2x-cli.py \
    -a 3 -e 20 -b 40 -k 3 -H 250 -m 350 -K 10 -r 10 -M aimle \
    -c models/beeradv_H=250_K=10_M=aimle_a=3_aisym=True_aitgt=adaptive_b=40_bm=0.0_bu=0.0001_e=20_ilmd=0.0_inoise=gumbel_ismp=1_itmp=1.0_k=3_m=350_r=10_scale=True_sst_temp=0.0_ssub_tmp=0.5_ste_noise=sog_ste_tmp=0.0_tn=1.0.pt \
    --aimle-symmetric --aimle-target adaptive --imle-samples 1 --imle-noise gumbel \
    --imle-input-temperature 1.0 --imle-output-temperature 1.0 --imle-lambda 0.0 \
    --sst-temperature 0.0 --softsub-temperature 0.5 --ste-noise sog --ste-temperature 0.0 \
    --gradient-scaling --aimle-beta-update-momentum 0.0 --aimle-beta-update-step 0.0001 \
    --aimle-target-norm 1.0
```

In the former command line, `-M aimle` specifies to use AIMLE; `--aimle-symmetric` specifies to use the Central Difference variant; and `--aimle-target adaptive` to use the adaptive target distribution where the perturbation size `\lambda` is selected automatically (`--aimle-target standard` uses the classic IMLE target distribution, which corresponds to using IMLE).

## Discrete Variational Auto-Encoders

Running the experiments:

```bash
$ WANDB_MODE=offline PYTHONPATH=. python3 ./cli/torch-vae-cli.py \
    -e 100 -b 100 -K 10 --code-m 20 --code-n 20 -M aimle \
    --imle-samples 10 --imle-noise sog --imle-temperature 1.0 --imle-lambda 0.0 \
    --aimle-symmetric --aimle-target adaptive --ste-noise sog --ste-temperature 0.0 \
    --seed 0 --init-temperature 0.0 --gradient-scaling \
    --aimle-beta-update-momentum 0.0 --aimle-beta-update-step 0.001 --aimle-target-norm 10.0

INFO:torch-vae-cli.py:Epoch 1/100       Training Loss: 1102.3853 ± 493.7140     Temperature: 0.00000    Beta: 0.60000
INFO:torch-vae-cli.py:Average Test Loss: 79.85607
INFO:torch-vae-cli.py:Epoch 2/100       Training Loss: 765.9837 ± 30.3491       Temperature: 0.50000    Beta: 1.05800
INFO:torch-vae-cli.py:Average Test Loss: 72.88469
INFO:torch-vae-cli.py:Epoch 3/100       Training Loss: 720.9445 ± 20.9125       Temperature: 0.50000    Beta: 1.05600
INFO:torch-vae-cli.py:Average Test Loss: 70.79888
INFO:torch-vae-cli.py:Epoch 4/100       Training Loss: 699.9427 ± 20.1001       Temperature: 0.50000    Beta: 1.06400
[..]
INFO:torch-vae-cli.py:Epoch 99/100	Training Loss: 593.7420 ± 14.8554	Temperature: 0.50000	Beta: 0.29800
INFO:torch-vae-cli.py:Average Test Loss: 60.44106
INFO:torch-vae-cli.py:Epoch 100/100	Training Loss: 593.5313 ± 15.2714	Temperature: 0.50000	Beta: 0.27000
INFO:torch-vae-cli.py:Average Test Loss: 60.20955
INFO:torch-vae-cli.py:Experiment completed.
wandb: Waiting for W&B process to finish... (success).
wandb:                                                                                
wandb: 
wandb: Run history:
wandb:      loss █▃▂▂▂▂▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb:       tau ▁███████████████████████████████████████
wandb: test_loss █▅▃▃▃▃▂▂▂▂▂▂▂▂▂▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb: 
wandb: Run summary:
wandb:      loss 593.5313
wandb:       tau 0.5
wandb: test_loss 60.20955
```

## Neural Relational Inference

For training the AIMLE models with different configurations (Central vs Forward, T=10 vs T=20, etc.) using 10 seeds each, get the list of command lines via the following command:

```bash
$ ./experiments/generate_nri_aimle_cmd.py
```

Each experiment will look like the following:

```bash
$ PYTHONPATH=. python3 ./cli/torch-nri-cli.py --suffix _novar_1skip_10t_1r_graph10 --timesteps 10 \
--prediction_steps 9 --sst tree --relaxation exp_family_entropy --max_range 15 --symmeterize_logits True \
--lr 0.0001 --temp 0.1 --eps_for_finitediff 1.0 --cuda True \
--experiments_folder ../exp_nri/nri_T=10_bu=0.001_eps=1.0_hard=True_imle_samples=1_lmbda=0.0_lr=0.0001_method=aimle_noise=sog_noise_temp=1.0_scaling=False_sst=tree_symmetric=False_target=adaptive_temp=0.1 \
--use_cpp_for_sampling True --method aimle --imle-lambda 0.0 --imle-lambda-update-step 0.001 --imle-noise sog \
--imle-noise-temperature 1.0 --aimle-target adaptive --imle-samples 1 --hard True --st True \
--experiments_folder ../exp_nri/aimle_v3_T10_nosym_best_3 --seed 3

Namespace(add_timestamp=True, aimle_symmetric=False, aimle_target='adaptive', batch_size=128, cuda=False, dec_weight_decay=0.0, decoder_dropout=0.0, decoder_hidden=256, dims=2, edge_metric_num_samples=1, edge_types=2, ema_for_loss=0.99, enc_weight_decay=0.0, encoder_dropout=0.0, encoder_hidden=256, eps_for_finitediff=1.0, eval_batch_size=100, eval_edge_metric_bs=10000, eval_every=500, experiment_name=None, experiments_folder='../exp_nri/aimle_v3_T10_nosym_best_3', factor=True, gamma=0.5, gradient_scaling=False, hard=True, imle_lambda=0.0, imle_lambda_update_step=0.001, imle_noise='sog', imle_noise_temperature=1.0, imle_samples=1, log_edge_metric_train=False, log_edge_metric_val=True, lr=0.0001, lr_decay=200, max_range=15.0, max_steps=None, method='aimle', mode='eval', num_iterations=50000, num_rounds=1, num_samples=1, num_vertices=10, prediction_steps=9, reinforce_baseline='ema', relaxation='exp_family_entropy', save_best_model=True, seed=3, skip_first=False, sst='tree', st=True, suffix='_novar_1skip_10t_1r_graph10', symmeterize_logits=True, temp=0.1, timesteps=10, use_cpp_for_edge_metric=False, use_cpp_for_sampling=True, use_gumbels_for_kl=True, use_nvil=False, use_reinforce=False, var=5e-05, verbose=False)
Using factor graph MLP encoder.
Using learned interaction net decoder.
Using Adam.
trial_path 1 experiments/../exp_nri/aimle_v3_T10_nosym_best_3/lr0.0001_temp0.1_encwd0.0_decwd0.0_3 True
trial_path 2 experiments/../exp_nri/aimle_v3_T10_nosym_best_3/lr0.0001_temp0.1_encwd0.0_decwd0.0_3/train_and_val_measurements.pkl True
encoder MLPEncoder(
  (mlp1): MLP(
    (fc1): Linear(in_features=20, out_features=256, bias=True)
    (fc2): Linear(in_features=256, out_features=256, bias=True)
    (bn): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  )
  (mlp2): MLP(
    (fc1): Linear(in_features=512, out_features=256, bias=True)
    (fc2): Linear(in_features=256, out_features=256, bias=True)
    (bn): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  )
  (mlp3): MLP(
    (fc1): Linear(in_features=256, out_features=256, bias=True)
    (fc2): Linear(in_features=256, out_features=256, bias=True)
    (bn): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  )
  (mlp4): MLP(
    (fc1): Linear(in_features=768, out_features=256, bias=True)
    (fc2): Linear(in_features=256, out_features=256, bias=True)
    (bn): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  )
  (fc_out): Linear(in_features=256, out_features=1, bias=True)
)
decoder MLPDecoder(
  (msg_fc1): ModuleList(
    (0): Linear(in_features=4, out_features=256, bias=True)
    (1): Linear(in_features=4, out_features=256, bias=True)
  )
  (msg_fc2): ModuleList(
    (0): Linear(in_features=256, out_features=256, bias=True)
    (1): Linear(in_features=256, out_features=256, bias=True)
  )
  (out_fc1): Linear(in_features=258, out_features=256, bias=True)
  (out_fc2): Linear(in_features=256, out_features=256, bias=True)
  (out_fc3): Linear(in_features=256, out_features=2, bias=True)
)

[..]

valid trial 0 for ../exp_nri/aimle_v3_T10_nosym_best_3 took 50.58802604675293s.
test trial 0 for ../exp_nri/aimle_v3_T10_nosym_best_3 took 103.33455300331116s.
all_measurements {'valid': {'elbo': array([-1660.79718872]), 'acc': array([0.8550089], dtype=float32), 'precision': array([0.6375222], dtype=float32), 'recall': array([0.6375222], dtype=float32)}, 'test': {'elbo': array([-1659.31648804]), 'acc': array([0.8546978], dtype=float32), 'precision': array([0.63674444], dtype=float32), 'recall': array([0.63674444], dtype=float32)}}
Saving data.
```

## Citing this work

```bibtex
@inproceedings{minervini23aimle,
  author    = {Pasquale Minervini and
               Luca Franceschi and
               Mathias Niepert},
  title     = {Adaptive Perturbation-Based Gradient Estimation for Discrete Latent Variable Models},
  booktitle = {{AAAI}},
  publisher = {{AAAI} Press},
  year      = {2023}
}
```
