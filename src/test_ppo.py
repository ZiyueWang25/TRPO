#!/usr/bin/env python3

import argparse
import sys

import torch

sys.path.append("../tianshou/")
sys.path.append("../tianshou/examples/mujoco/")
sys.path.append("../tianshou/examples/atari/")
from mujoco_ppo import test_ppo as test_ppo_mujoco
from mujoco_ppo import get_args as get_args_mujoco
from atari_ppo import test_ppo as test_ppo_atari
from atari_ppo import get_args as get_args_atari

class PPO_MUJOCO_Base:
    task = "Hopper-v3"
    seed = 0
    buffer_size = 4096
    hidden_sizes = [64, 64]
    lr = 3e-4
    gamma = 0.99
    epoch = 100
    step_per_epoch = int(3e5)
    step_per_collect = 2048
    repeat_per_collect = 10
    batch_size = 64
    training_num = 64
    test_num = 10
    rew_norm = True # ppo special
    vf_coef = 0.25 # in theory, "vf-coef" won't make any difference if using Adam Optimizer
    ent_coef = 0 
    gae_lambda = 0.95
    bound_action_method = 'clip'
    lr_decay = True
    max_grad_norm = 0.5
    eps_clip = 0.2
    dual_clip = None
    value_clip = 0
    norm_adv = 0
    recompute_adv = 1
    logdir = "../log/"
    render = 0
    device = "cuda"
    resume_path = None
    watch = False

MUJOCO_GAME_DICT = {
    "swimmer": "Swimmer-v3",
    "hopper": "Hopper-v3",
    "walker": "Walker2d-v3"
}

ATARI_GAME_DICT = {
    "breakout" : "BreakoutNoFrameskip-v4",
    "pong": "PongNoFrameskip-v4",
    "enduro": "EnduroNoFrameskip-v4",
    "qbert": "QbertNoFrameskip-v4",
    "seaquest": "SeaquestNoFrameskip-v4",
    "spaceinvaders": "SpaceInvadersNoFrameskip-v4",
    "mspacman": "MsPacmanNoFrameskip-v4"
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='swimmer')
    parser.add_argument(
        '--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu'
    )
    parser.add_argument('--logdir', type=str, default='../log')    
    u_args = parser.parse_args()
    if u_args.task in MUJOCO_GAME_DICT.keys():
        test_fnc = test_ppo_mujoco
        get_args = get_args_mujoco
    else:
        test_fnc = test_ppo_atari
        get_args = get_args_atari
        
    args = get_args()        
    args.task = MUJOCO_GAME_DICT[u_args.task] if u_args.task in MUJOCO_GAME_DICT else ATARI_GAME_DICT[u_args.task]
    args.logdir = u_args.logdir
    args.device = u_args.device
    test_fnc(args)