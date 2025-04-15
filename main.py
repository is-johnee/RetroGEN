#!/usr/bin/env python3

import os
import random

import numpy as np
import torch
from tensorboardX import SummaryWriter
from ppo_motif import ppo
import gym
from core_motif_vbased import GATActorCritic
from mpi_tools import mpi_fork
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def train(args, writer=None):
    set_seed(args.seed)
    env = gym.make('molecule-v0')
    env.seed(args.seed)
    env.init(docking_config=args.docking_config, ratios=args.ratios, reward_step_total=args.reward_step_total,
             is_normalize=args.normalize_adj, has_feature=bool(args.has_feature), max_action=args.max_action,
             min_action=args.min_action)

    if args.n_cpus > 1:
        mpi_fork(args.n_cpus)

    PPO = ppo(writer, args, env, actor_critic=GATActorCritic, ac_kwargs=dict(), seed=args.seed,
              steps_per_epoch=args.steps_per_epoch, epochs=1000, replay_size=int(1e6), gamma=0.99,
              polyak=0.995, lr=args.init_lr, alpha=args.init_alpha, batch_size=args.batch_size,
              num_test_episodes=8, train_alpha=True)
    PPO.train()


def arg_parser():
    """
    Create an empty argparse.ArgumentParser.
    """
    import argparse
    return argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)


def molecule_arg_parser():
    parser = arg_parser()

    # Choose RL model
    parser.add_argument('--rl_model', type=str, default='ppo')
    parser.add_argument('--molecule_save_path', type=str, default='molecule_gen/')
    parser.add_argument('--device', type=int, default='0')
    parser.add_argument('--train', type=int, default=1, help='training or inference')
    parser.add_argument('--resume', type=int, default=0, help='resume or from beginning')
    # env
    parser.add_argument('--env', type=str, help='environment name: molecule; graph', default='molecule')
    parser.add_argument('--seed', help='RNG seed', type=int, default=42)  # 2023,2022,2021,2024,2025
    parser.add_argument('--num_steps', type=int, default=int(5e7))

    parser.add_argument('--model_path', type=str, default='')
    parser.add_argument('--reward_step_total', type=float, default=0.5)
    parser.add_argument('--target', type=str, default='5uk8', help='5uk8, 4bci, cdk9')

    parser.add_argument('--tau', type=float, default=1)

    # model update
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--init_lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--update_every', type=int, default=256)
    parser.add_argument('--update_freq', type=int, default=256)
    parser.add_argument('--update_after', type=int, default=200)
    parser.add_argument('--start_steps', type=int, default=3000)

    # model save and load
    parser.add_argument('--save_every', type=int, default=500)
    parser.add_argument('--load_step', type=int, default=250)

    # graph embedding
    parser.add_argument('--gnn_type', type=str, default='gat')
    parser.add_argument('--gin_aggregate', type=str, default='sum')
    # parser.add_argument('--graph_emb', type=int, default=0)
    parser.add_argument('--emb_dim', type=int, default=128)  # default 64
    parser.add_argument('--bond_dim', type=int, default=32)
    parser.add_argument('--has_residual', type=int, default=0)
    parser.add_argument('--has_feature', type=int, default=1)

    parser.add_argument('--normalize_adj', type=int, default=0)
    parser.add_argument('--bn', type=int, default=0)

    parser.add_argument('--layer_num_g', type=int, default=4)

    parser.add_argument('--max_action', type=int, default=4)
    parser.add_argument('--min_action', type=int, default=1)

    parser.add_argument('--pretrained_model_path', type=str, default='ckpt\pretrained_model.pth')

    parser.add_argument('--init_alpha', type=float, default=1.)

    parser.add_argument('--n_cpus', type=int, default=1)
    parser.add_argument('--steps_per_epoch', type=int, default=257)

    return parser


def get_docking_config():
    docking_config = dict()
    docking_config['vina_program'] = 'vina'
    docking_config['exhaustiveness'] = 4
    docking_config['num_sub_proc'] = 10
    docking_config['num_cpu_dock'] = 5
    docking_config['num_modes'] = 10
    docking_config['timeout_gen3d'] = 30
    docking_config['timeout_dock'] = 100
    return docking_config

def get_ratios():
    ratios = dict()
    ratios['logp'] = 0
    ratios['qed'] = 1
    ratios['sa'] = 1
    ratios['mw'] = 0
    ratios['filter'] = 0
    ratios['docking'] = 1
    return ratios

def main():
    args = molecule_arg_parser().parse_args()
    docking_config = get_docking_config()
    ratios = get_ratios()

    assert args.target in ['5uk8', '4bci'], "Wrong target type"
    if args.target == '5uk8':
        box_center = (-2.931, -8.697, 19.805)
        box_size = (15, 15, 15)
        docking_config['receptor_file'] = 'docking_target\5uk8\receptor.pdbqt'
        docking_config['temp_dir'] = '5uk8_tmp'
    elif args.target == '4bci':
        box_center = (56.201, -16.703, -12.863)
        box_size = (15, 15, 15)
        docking_config['receptor_file'] = 'docking_target\4bci\receptor.pdbqt'
        docking_config['temp_dir'] = '4bci_tmp'
    box_parameter = (box_center, box_size)
    docking_config['box_parameter'] = box_parameter
    
    if args.train:
        docking_config['exhaustiveness'] = 4
    else:
        docking_config['exhaustiveness'] = 10


    args.docking_config = docking_config
    args.ratios = ratios

    # check and clean
    if not os.path.exists(args.molecule_save_path):
        os.makedirs(args.molecule_save_path)
    if not os.path.exists('ckpt'):
        os.makedirs('ckpt')
    if not os.path.exists('runs'):
        os.makedirs('runs')
    writer = SummaryWriter('runs/' + args.target)

    train(args, writer=writer)


if __name__ == '__main__':
    main()
