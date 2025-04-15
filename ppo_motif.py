# Copyright 2021 AITRICS [and/or other original copyright holders]
# Copyright 2025 Yi
#
# This file is modified from https://github.com/AITRICS/[specific-repository]
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
from copy import deepcopy
import itertools

import numpy as np
from rdkit import Chem

import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam, lr_scheduler

import gym

import core_motif_vbased as core
from gym_molecule.envs.env_utils_graph import FRAG_VOCAB, ATOM_VOCAB

from mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs


def delete_multiple_element(list_object, indices):
    indices = sorted(indices, reverse=True)
    for idx in indices:
        if idx < len(list_object):
            list_object.pop(idx)


def get_att_points(mol):
    att_points = []
    for a in mol.GetAtoms():
        if a.GetSymbol() == '*':
            att_points.append(a.GetIdx())
    return att_points


def get_final_smi(smi):
    # m = Chem.ReplaceSubstructs(Chem.MolFromSmiles(smi), Chem.MolFromSmiles("*"))
    m = Chem.ReplaceSubstructs(Chem.MolFromSmiles(smi), Chem.MolFromSmiles('*'), Chem.MolFromSmiles('[H]'),
                               replaceAll=True)
    Chem.SanitizeMol(m[0], sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
    return Chem.MolToSmiles(m[0])


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = []  # o
        self.obs2_buf = []  # o2
        self.act_buf = np.zeros((size, 2), dtype=np.int32)  # ac
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)  # r
        self.ret_buf = np.zeros(size, dtype=np.float32)  # v
        self.val_buf = np.zeros(size, dtype=np.float32)  # v
        self.logp_buf = np.zeros((size, 2), dtype=np.float32)  # v
        self.done_buf = np.zeros(size, dtype=np.float32)  # d

        self.ac_prob_buf = []
        self.log_ac_prob_buf = []

        self.ac_first_buf = []
        self.ac_second_buf = []
        self.ac_third_buf = []

        self.o_embeds_buf = []

        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size
        self.tmp_size = self.max_size

    def store(self, obs, next_obs, act, rew, val, logp, done):
        assert self.ptr < self.max_size  # buffer has to have room so you can store

        self.obs_buf.append(obs)
        self.obs2_buf.append(next_obs)

        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.done_buf[self.ptr] = done

        self.ptr += 1

    def rew_store(self, rew, batch_size=32):
        rew_ls = list(rew)

        done_location_np = np.array(self.done_location)
        zeros = np.where(rew == 0.0)[0]
        nonzeros = np.where(rew != 0.0)[0]
        zero_ptrs = done_location_np[zeros]

        done_location_np = done_location_np[nonzeros]
        rew = rew[nonzeros]

        if len(self.done_location) > 0:
            self.rew_buf[done_location_np] += rew
            self.done_location = []

        self.act_buf = np.delete(self.act_buf, zero_ptrs, axis=0)
        self.rew_buf = np.delete(self.rew_buf, zero_ptrs)
        self.done_buf = np.delete(self.done_buf, zero_ptrs)
        delete_multiple_element(self.obs_buf, zero_ptrs.tolist())
        delete_multiple_element(self.obs2_buf, zero_ptrs.tolist())

        self.size = min(self.size - len(zero_ptrs), self.max_size)
        self.ptr = (self.ptr - len(zero_ptrs)) % self.max_size

    def sample_batch(self, device, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        obs_batch = [self.obs_buf[idx] for idx in idxs]
        obs2_batch = [self.obs2_buf[idx] for idx in idxs]

        act_batch = torch.as_tensor(self.act_buf[idxs], dtype=torch.float32).unsqueeze(-1).to(device)
        rew_batch = torch.as_tensor(self.rew_buf[idxs], dtype=torch.float32).to(device)
        done_batch = torch.as_tensor(self.done_buf[idxs], dtype=torch.float32).to(device)

        batch = dict(obs=obs_batch,
                     obs2=obs2_batch,
                     act=act_batch,
                     rew=rew_batch,
                     done=done_batch)

        return batch

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.
        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = core.discount_cumsum(deltas, self.gamma * self.lam)

        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = core.discount_cumsum(rews, self.gamma)[:-1]

        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size  # buffer has to be full before you can get
        # self.max_size==args.steps_per_epoch
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std

        data = dict(obs=self.obs_buf,
                    act=torch.as_tensor(self.act_buf),
                    ret=torch.as_tensor(self.ret_buf, dtype=torch.float32),
                    adv=torch.as_tensor(self.adv_buf, dtype=torch.float32),
                    logp=torch.as_tensor(self.logp_buf, dtype=torch.float32))

        self.obs_buf = []
        self.obs2_buf = []
        return {k: v for k, v in data.items()}


class ppo:
    """
    
    """

    def __init__(self, writer, args, env_fn, actor_critic=core.GATActorCritic, ac_kwargs=dict(), seed=0,
                 steps_per_epoch=4000, epochs=100, replay_size=int(1e6), gamma=0.99,
                 polyak=0.995, lr=1e-3, alpha=0.2, batch_size=100, num_test_episodes=10, train_alpha=True):
        super().__init__()
        self.device = args.device

        torch.manual_seed(seed)
        np.random.seed(seed)
        self.gamma = gamma
        self.polyak = polyak
        self.num_test_episodes = num_test_episodes
        self.writer = writer
        self.fname = args.molecule_save_path + args.target + '_generated.csv'
        self.test_fname = args.molecule_save_path + args.target + '_test.csv'
        self.save_name = './ckpt/' + args.target
        self.steps_per_epoch = steps_per_epoch
        self.epochs = epochs
        self.batch_size = batch_size
        self.replay_size = replay_size

        self.train_alpha = train_alpha
        self.smis_trajectory = []
        self.lr = lr

        self.env, self.test_env = env_fn, deepcopy(env_fn)

        self.obs_dim = args.emb_dim * 2
        self.act_dim = len(FRAG_VOCAB) - 1

        self.ac2_dims = len(FRAG_VOCAB)  # 76
        self.ac3_dims = 40
        self.action_dims = [self.ac2_dims, self.ac3_dims]

        # On-policy

        self.train_pi_iters = 10
        self.train_v_iters = 10
        self.target_kl = 0.01
        self.steps_per_epoch = steps_per_epoch
        self.local_steps_per_epoch = steps_per_epoch // num_procs()
        self.epochs = epochs
        self.clip_ratio = .2
        self.ent_coeff = .01

        self.n_cpus = args.n_cpus

        self.target_entropy = 1.0

        self.log_alpha = torch.tensor([np.log(alpha)], requires_grad=train_alpha)
        alpha = self.log_alpha.exp().item()

        # Create actor-critic module and target networks
        self.ac = actor_critic(self.env, args).to(args.device)
        if args.train == 1:
            self.ac.pi.embed.load_state_dict(
                torch.load(args.pretrained_model_path))
            # for para in self.ac.pi.embed.parameters():
            #     para.requires_grad = False
        else:
            self.ac.load_state_dict(torch.load(args.name_full_load))
        self.ac_targ = deepcopy(self.ac).to(args.device).eval()

        # Sync params across processes
        sync_params(self.ac)
        for p in self.ac_targ.parameters():
            p.requires_grad = False

        for q in self.ac.parameters():
            q.requires_grad = True

        self.replay_buffer = ReplayBuffer(obs_dim=self.obs_dim, act_dim=self.act_dim, size=self.local_steps_per_epoch,
                                          gamma=1., lam=.95)

        # Count variables (protip: try to get a feel for how different size networks behave!)
        self.var_counts = tuple(core.count_vars(module) for module in [self.ac.pi, self.ac.v])
        self.iter_so_far = 0
        self.ep_so_far = 0

        ## OPTION1: LEARNING RATE
        pi_lr = self.lr
        vf_lr = self.lr

        ## OPTION2: OPTIMIZER SETTING        
        self.pi_params = list(self.ac.pi.parameters())
        self.vf_params = list(self.ac.v.parameters())

        self.pi_optimizer = Adam(self.pi_params, lr=pi_lr, weight_decay=1e-8)
        self.vf_optimizer = Adam(self.vf_params, lr=vf_lr, weight_decay=1e-8)

        self.vf_scheduler = lr_scheduler.ReduceLROnPlateau(self.vf_optimizer, factor=0.1, patience=768)
        self.pi_scheduler = lr_scheduler.ReduceLROnPlateau(self.pi_optimizer, factor=0.1, patience=768)

        self.L2_loss = torch.nn.MSELoss()

        torch.set_printoptions(profile="full")
        # self.possible_bonds = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE]

        self.t = 0

        tm = time.localtime(time.time())
        self.init_tm = time.strftime('_%Y_%m_%d_%I_%M_%S_%p', tm)

    def compute_loss_v(self, data):
        obs, ret = data['obs'], data['ret'].to(self.device)
        o_g, o_n_emb, o_g_emb = self.ac.pi.embed(obs)

        return ((self.ac.v(o_g_emb) - ret) ** 2).mean()

    def compute_loss_pi(self, data):
        obs, act, adv, logp_old = data['obs'], data['act'], \
                                  data['adv'].to(self.device).unsqueeze(1), data['logp'].to(self.device)
        with torch.no_grad():
            o_embeds = self.ac.pi.embed(data['obs'])
            o_g, o_n_emb, o_g_emb = o_embeds
            cands = self.ac.pi.embed(self.ac.pi.cand)
        ac, ac_prob, log_ac_prob, final_smiles_list, bond_idx_list = self.ac.pi(o_g_emb, o_n_emb, o_g, cands,
                                                                                data['obs'])
        dists = self.ac.pi._distribution(ac_prob)
        logp = self.ac.pi._log_prob_from_distribution(dists, ac)
        # ratio = torch.exp(logp.sum(1) - logp_old.sum(1))
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * adv
        # loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()
        loss_pi_action1 = -(torch.min(ratio * adv, clip_adv)[:, 0]).mean()
        loss_pi_action2 = -(torch.min(ratio * adv, clip_adv)[:, 1]).mean()
        return loss_pi_action1, loss_pi_action2

    def update(self):
        data = self.replay_buffer.get()
        for i in range(self.train_pi_iters):
            self.pi_optimizer.zero_grad()
            loss_pi_1, loss_pi_2 = self.compute_loss_pi(data)
            loss_pi = (loss_pi_1 + loss_pi_2).mean()
            loss_pi.backward()
            mpi_avg_grads(self.ac.pi)
            self.pi_optimizer.step()
            self.pi_scheduler.step(loss_pi)

        for i in range(self.train_v_iters):
            self.vf_optimizer.zero_grad()
            loss_v = self.compute_loss_v(data)
            loss_v.backward()
            mpi_avg_grads(self.ac.v)  # average grads across MPI processes
            self.vf_optimizer.step()
            self.vf_scheduler.step(loss_v)

        # Record things
        if proc_id() == 0:
            if self.writer is not None:
                iter_so_far_mpi = self.iter_so_far * self.n_cpus
                self.writer.add_scalar("loss_V", loss_v.item(), iter_so_far_mpi)
                self.writer.add_scalar("loss_Pi", loss_pi.item(), iter_so_far_mpi)

    def get_action(self, o, deterministic=False):
        return self.ac.act(o, deterministic)

    def train(self):
        total_steps = self.steps_per_epoch * self.epochs
        start_time = time.time()
        self.iter_so_far = 0
        o, ep_ret, ep_len = self.env.reset(), 0, 0

        for epoch in range(self.epochs):
            for t in range(self.local_steps_per_epoch):
                self.t = t
                # with torch.no_grad():
                cands = self.ac.pi.embed(self.ac.pi.cand)
                o_g, o_n_emb, o_g_emb = self.ac.pi.embed([o])
                ac, v, logp, final_smiles_list, bond_idx_list = self.ac.step(o_g_emb, o_n_emb, o_g, cands,
                                                                             [o])
                print(ac)

                o2, r, d, info = self.env.step(ac, final_smiles_list, bond_idx_list)  # step-wise reward
                r_d = info['stop']

                # Store experience to replay buffer

                if type(ac) == np.ndarray:
                    self.replay_buffer.store(o, o2, ac, r, v, logp, r_d)
                else:
                    self.replay_buffer.store(o, o2, ac.detach().cpu().numpy(), r, v, logp, r_d)

                # Super critical, easy to overlook step: make sure to update 
                # most recent observation!
                o = o2

                # End of trajectory handling
                if get_att_points(self.env.mol) == []:  # Temporally force attachment calculation
                    d = True
                if not any(o2['att']):
                    d = True
                if d:
                    final_smi = get_final_smi(o2['smi'])
                    # sascore = 10 - self.env.reward_single(
                    #     [final_smi])
                    docking_score = self.env.reward_single(
                        [final_smi])

                    if docking_score[0] > 0:
                        iter_so_far_mpi = self.iter_so_far * self.n_cpus
                        if proc_id() == 0:
                            # self.writer.add_scalar("EpRet", ext_rew[0], iter_so_far_mpi)
                            self.writer.add_scalar("EpSA", docking_score[0], iter_so_far_mpi)
                            # self.writer.add_scalar("TotalRet", ext_rew[0] + sascore, iter_so_far_mpi)
                        with open(self.fname[:-5] + self.init_tm + '_47_4_16_freq_10.csv', 'a') as f:
                            mol_info = f'{final_smi},{docking_score[0]},{iter_so_far_mpi}' + '\n'
                            f.write(mol_info)
                    # norm_sascore = round((10-sascore)/9,2)
                    # self.replay_buffer.finish_path(ext_rew - sascore)
                    self.replay_buffer.finish_path(docking_score[0])
                    o, ep_ret, ep_len = self.env.reset(), 0, 0
                    self.env.smile_list = []
                    self.ep_so_far += 1
                self.iter_so_far += 1

            t_update = time.time()
            print('updating model...')
            # if epoch == 2:
            #     self.update()
            # else:
            self.update()
            dt_update = time.time()
            print('update time : ', t, dt_update - t_update)
            if epoch % 1 == 0 and epoch != 0:
                fname = self.save_name + f'{epoch}'
                torch.save(self.ac.state_dict(), fname + '_atr_frags_rewrite.pt')
                print('model saved!', fname)

    def test(self):
        num_generated = 0
        self.iter_so_far = 0
        max_num = 20000
        o, ep_ret, ep_len = self.env.reset(), 0, 0
        self.smis_trajectory.append(get_final_smi(self.env.smi))
        for epoch in range(self.epochs):
            if num_generated >= max_num:
                break
            for t in range(self.local_steps_per_epoch):
                with torch.no_grad():
                    cands = self.ac.pi.embed(self.ac.pi.cand)
                    o_embeds = self.ac.pi.embed([o])
                    o_g, o_n_emb, o_g_emb = o_embeds
                    ac, v, logp, final_smiles_list, bond_idx_list = self.ac.step(o_g_emb, o_n_emb, o_g, cands,
                                                                                 [o])
                    print(ac)
                o2, r, d, info = self.env.step(ac, final_smiles_list, bond_idx_list)
                self.smis_trajectory.append(get_final_smi(o2['smi']))
                # Super critical, easy to overlook step: make sure to update
                # most recent observation!
                o = o2
                # End of trajectory handling
                if get_att_points(self.env.mol) == []:  # Temporally force attachment calculation
                    d = True
                if not any(o2['att']):
                    d = True
                if d:
                    final_smi = get_final_smi(o2['smi'])
                    with open(self.fname[:-5] + self.init_tm + '_5uk8_150_seed_42_full_3_12_feq5_max3_20000.csv', 'a') as f:
                        mol_info = f'{final_smi}'
                        for i in range(len(self.smis_trajectory) - 1):
                            mol_info += ','
                            mol_info += self.smis_trajectory[i]
                        mol_info += '\n'
                        f.write(mol_info)
                        self.smis_trajectory.clear()
                    o, ep_ret, ep_len = self.env.reset(), 0, 0
                    self.smis_trajectory.append(get_final_smi(self.env.smi))
                    self.env.smile_list = []
                    self.ep_so_far += 1
                    num_generated = num_generated + 1
                    if num_generated >= max_num:
                        break
                self.iter_so_far += 1
