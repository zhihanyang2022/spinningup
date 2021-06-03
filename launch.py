import gym
import sys
import wandb
import os
sys.path.append("../off-policy-continuous-control/offpcc")
from domains import *
import torch.nn as nn

wandb.init(
    project=os.getenv('OFFPCC_WANDB_PROJECT'),
    entity=os.getenv('OFFPCC_WANDB_ENTITY'),
    group=f"car-v0 ddpg (spinup)",
    settings=wandb.Settings(_disable_stats=True),
    name=f'run_id=1',
    reinit=True
)

from spinup import ddpg_pytorch as ddpg


ac_kwargs = dict(hidden_sizes=[256, 256, 256], activation=nn.ReLU)

logger_kwargs = dict(output_dir='data/car_ddpg', exp_name='car_ddgp')

ddpg(
    env_fn=lambda: gym.make('car-concat20-v0'),
    ac_kwargs=ac_kwargs,
    update_every=1,
    update_after=999,
    start_steps=999,
    steps_per_epoch=1000,
    epochs=200,
    logger_kwargs=logger_kwargs,
    max_ep_len=160
)
