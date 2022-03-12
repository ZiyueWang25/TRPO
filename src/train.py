#!/usr/bin/env python3

import argparse
import sys
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

import torch

sys.path.append("../tianshou/")
sys.path.append("../tianshou/examples/mujoco/")
sys.path.append("../tianshou/examples/atari/")
import mujoco_ppo
import mujoco_trpo
import atari_ppo

class PPO_MUJOCO:
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
    
class TRPO_MUJOCO(PPO_MUJOCO):
    # batch-size >> step-per-collect means calculating all data in one singe forward.
    batch_size = 99999
    training_num = 16
    # trpo special
    rew_norm = True
    gae_lambda = 0.95
    optim_critic_iters = 20
    max_kl = 0.01
    backtrack_coeff = 0.8
    max_backtracks = 10
    
class TRPO_MUJOCO_Swimmer(TRPO_MUJOCO):
    step_per_epoch = int(5e4)
    hidden_sizes = [30]
    gamma = 0.99
    

POLICY_DICT = {
    "ppo": {
        "mujoco": (mujoco_ppo.test_ppo, mujoco_ppo.get_args),
        "atari": (atari_ppo.test_ppo, atari_ppo.get_args),
    },
    "trpo": {
        "mujoco": (mujoco_trpo.test_trpo, mujoco_trpo.get_args),
    }
}

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
    parser.add_argument('--method', type=str, default='ppo')
    parser.add_argument('--task', type=str, default='swimmer')
    parser.add_argument(
        '--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu'
    )
    parser.add_argument('--logdir', type=str, default='../log')    
    u_args = parser.parse_args()
    method = u_args.method
    if u_args.task in MUJOCO_GAME_DICT.keys():
        test_fnc, get_args = POLICY_DICT[method]["mujoco"]
        task = MUJOCO_GAME_DICT[u_args.task]
    else:
        test_fnc, get_args = POLICY_DICT[method]["atari"]
        task = ATARI_GAME_DICT[u_args.task]
    sys.argv = ["-m"] # to avoid passing the system args down the line
    args = get_args()        
    args.task = task
    args.logdir = u_args.logdir
    args.device = u_args.device
    test_fnc(args)