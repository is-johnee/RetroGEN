"""
Copyright 2021 AITRICS [and/or other original copyright holders]
Copyright 2025 Yi

This file is modified from https://github.com/AITRICS/[specific-repository]

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import scipy.signal
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from model import GNN_graphCL
from loader import mol_to_graph_data

from rdkit import Chem
from gym_molecule.envs.env_utils_graph import ATOM_VOCAB, FRAG_VOCAB, FRAG_VOCAB_MOL


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


def get_att_points(mol):
    att_points = []
    for a in mol.GetAtoms():
        if a.GetSymbol() == '*':
            att_points.append(a.GetIdx())

    return att_points


def reduce_mean(nodes):
    accum = torch.mean(nodes.mailbox['m'], 1)
    return {'x': accum}


def reduce_sum(nodes):
    accum = torch.sum(nodes.mailbox['m'], 1)
    return {'x': accum}


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.
    input: 
        vector x, 
        [x0, 
         x1, 
         x2]
    output:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class GATActorCritic(nn.Module):
    def __init__(self, env, args):
        super().__init__()
        ob_space = env.observation_space
        ac_space = env.action_space
        self.env = env
        self.pi = SFSPolicy(ob_space, ac_space, env, args)
        self.v = GCNVFunction(ac_space, args)
        self.cand = self.create_candidate_motifs()

    def create_candidate_motifs(self):
        motif_gs = [self.env.get_observation_mol(mol) for mol in FRAG_VOCAB_MOL]
        return motif_gs

    def step(self, o_g_emb, o_n_emb, o_g, cands, o):
        with torch.no_grad():
            ac, ac_prob, log_ac_prob, final_smiles_list, bond_idx_list = self.pi(o_g_emb, o_n_emb, o_g, cands, o)
            dists = self.pi._distribution(ac_prob.cpu())
            logp_a = self.pi._log_prob_from_distribution(dists, ac.cpu())
            v = self.v(o_g_emb)

        return ac.cpu().numpy(), v.cpu().numpy(), logp_a.cpu().numpy(), final_smiles_list, bond_idx_list

    def act(self, o_g_emb, o_n_emb, o_g, cands):
        return self.step(o_g_emb, o_n_emb, o_g, cands)[0]

class GCNVFunction(nn.Module):
    def __init__(self, ac_space, args, override_seed=False):
        super().__init__()
        if override_seed:
            seed = args.seed + 1
            torch.manual_seed(seed)
            np.random.seed(seed)
        self.batch_size = args.batch_size
        self.emb_dim = args.emb_dim
        self.max_action2 = len(ATOM_VOCAB)
        self.max_action_stop = 2
        self.out_dim = 1
        self.vpred_layer = nn.Sequential(
            nn.Linear(self.emb_dim, int(self.emb_dim // 2), bias=True),
            nn.ReLU(inplace=False),
            nn.Linear(int(self.emb_dim // 2), self.out_dim, bias=True))

    def forward(self, o_g_emb):
        qpred = self.vpred_layer(o_g_emb)
        return qpred


# LOG_STD_MAX = 2
# LOG_STD_MIN = -20


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)


class SFSPolicy(nn.Module):
    def __init__(self, ob_space, ac_space, env, args):
        super().__init__()
        self.batch_size = args.batch_size
        self.ac_dim = len(FRAG_VOCAB) - 1
        self.emb_dim = args.emb_dim
        self.tau = args.tau
        # init candidate atoms
        self.bond_type_num = 4
        self.embed = GNN_graphCL(args.layer_num_g, args.device, args.emb_dim, args.bond_dim, drop_ratio=0.6,
                                 gnn_type=args.gnn_type)
        self.env = env  # env utilized to init cand motif mols
        self.cand = self.create_candidate_motifs()
        # Create candidate descriptors
        self.motif_type_num = len(self.cand)
        # self.device = args.device
        self.action_motif = nn.ModuleList(
            [nn.Bilinear(args.emb_dim, args.emb_dim, args.emb_dim),
             nn.Linear(args.emb_dim, args.emb_dim),
             nn.Linear(args.emb_dim, args.emb_dim),
             nn.Sequential(nn.Linear(args.emb_dim, args.emb_dim),
                           nn.ReLU(inplace=False),
                           nn.Linear(args.emb_dim, 1))])
        self.action_binding = nn.ModuleList(
            [nn.Bilinear(args.emb_dim, args.bond_dim, args.emb_dim),
             nn.Sequential(nn.Linear(args.emb_dim, args.emb_dim),
                           nn.ReLU(inplace=False),
                           nn.Linear(args.emb_dim, args.emb_dim),
                           nn.ReLU(inplace=False),
                           nn.Linear(args.emb_dim, 1))])
        self.max_action = 40  # max atoms
        print('number of candidate motifs : ', len(self.cand))

    def create_candidate_motifs(self):
        motif_gs = [self.env.get_observation_mol(mol) for mol in FRAG_VOCAB_MOL]
        # data_loader = DataLoader(motif_gs, batch_size=len(motif_gs))
        # batch_loader = next(iter(data_loader))
        return motif_gs

    def gumbel_softmax(self, logits: torch.Tensor, tau: float = 1, hard: bool = False, eps: float = 1e-10,
                       dim: int = -1, \
                       g_ratio: float = 1e-3) -> torch.Tensor:
        gumbels = (
            -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
        )  # ~Gumbel(0,1)
        gumbels = (logits + gumbels * g_ratio) / tau  # ~Gumbel(logits,tau)
        y_soft = gumbels.softmax(dim)
        dist = Categorical(y_soft)
        if hard:
            index = dist.sample().unsqueeze(1)
            y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
            ret = y_hard - y_soft.detach() + y_soft
        else:
            # Reparametrization trick.
            ret = y_soft
        return ret

    def forward(self, graph_emb, node_emb, g, cands, o, deterministic=False):
        """
        graph_emb : bs x hidden_dim
        node_emb : (bs x num_nodes) x hidden_dim)
        g: batched graph
        att: indexs of attachment points, list of list
        """
        g.node_emb = node_emb
        cand_g, cand_node_emb, cand_graph_emb = cands
        if (g.num_graphs != 1):
            graph_emb_motif = graph_emb.view(-1, 1, self.emb_dim).repeat(1, self.motif_type_num, 1)
            cand_expand = cand_graph_emb.unsqueeze(0).repeat(g.num_graphs, 1, 1)
            emb_first = self.action_motif[0](graph_emb_motif, cand_expand) + \
                        self.action_motif[1](graph_emb_motif) + self.action_motif[2](cand_expand)
            logit_motif = self.action_motif[3](emb_first).squeeze(-1)
            ac_motif_prob = F.softmax(logit_motif, dim=-1) + 1e-8

            log_ac_motif_prob = ac_motif_prob.log()
            ac_motif_hot = self.gumbel_softmax(ac_motif_prob, tau=self.tau, hard=True, g_ratio=1e-3, dim=1)
            ac_first = torch.argmax(ac_motif_hot, dim=-1)

            fragment_smi_list = [self.cand[ac_motif_i]['smi'] for ac_motif_i in ac_first]
            fragment_mol_list = [Chem.MolFromSmiles(smi) for smi in fragment_smi_list]
            starting_smi_list = [x['smi'] for x in o]
            starting_mol_list = [Chem.MolFromSmiles(smi) for smi in starting_smi_list]
            start_idx_list = [get_att_points(mol) for mol in starting_mol_list]
            end_idx_list = [get_att_points(mol) for mol in fragment_mol_list]
            final_mol_list = []
            final_smiles_list = []
            bond_idx_list = []
            for i, starting_mol in enumerate(starting_mol_list):
                _mol = []
                mol_tmp = []
                smiles_list = []
                bond_idx = []
                for idx_begin in range(len(start_idx_list[i])):
                    starting_mol_local = copy.deepcopy(starting_mol)
                    neighbors = starting_mol_local.GetAtomWithIdx(start_idx_list[i][idx_begin]).GetNeighbors()
                    for neighbor in neighbors:
                        neighbor.SetProp("Neighbors", "possible_binding_site")
                    starting_mol_local = Chem.RWMol(starting_mol_local)
                    starting_mol_local.RemoveAtom(start_idx_list[i][idx_begin])

                    for idx_end in range(len(end_idx_list[i])):
                        fragment_mol_local = copy.deepcopy(fragment_mol_list[i])
                        frag_neighbors = fragment_mol_local.GetAtomWithIdx(end_idx_list[i][idx_end]).GetNeighbors()
                        for frag_neighbor in frag_neighbors:
                            frag_neighbor.SetProp("Neighbors", "possible_binding_site")
                        fragment_mol_local = Chem.RWMol(fragment_mol_local)
                        fragment_mol_local.RemoveAtom(end_idx_list[i][idx_end])
                        comb_mol = Chem.CombineMols(starting_mol_local, fragment_mol_local)
                        idx = []
                        for atom in comb_mol.GetAtoms():
                            try:
                                if atom.GetProp('Neighbors') == 'possible_binding_site':
                                    idx.append(atom.GetIdx())
                            except:
                                continue
                        ed_mol = Chem.EditableMol(comb_mol)
                        ed_mol.AddBond(idx[0], idx[1], order=Chem.rdchem.BondType.SINGLE)
                        idx = []
                        mol_single = ed_mol.GetMol()
                        for atom in mol_single.GetAtoms():
                            atom.SetAtomMapNum(atom.GetIdx())
                        for atom in mol_single.GetAtoms():
                            try:
                                if atom.GetProp('Neighbors') == 'possible_binding_site':
                                    idx.append(atom.GetAtomMapNum())
                            except:
                                continue
                        mol_smi = Chem.MolToSmiles(mol_single)
                        smiles_list.extend([mol_smi])
                        mol_single = Chem.MolFromSmiles(mol_smi)
                        idx_set = set(idx)
                        for bond in mol_single.GetBonds():
                            if bond.GetBeginAtom().GetAtomMapNum() in idx_set and bond.GetEndAtom().GetAtomMapNum() in idx_set:
                                bond_idx.extend([bond.GetIdx()])
                        mol_tmp.extend([mol_single])
                bond_idx_list.append(bond_idx)
                final_smiles_list.append(smiles_list)
                final_mol_list.append(mol_tmp)
            graph_list = []
            for i, mol in enumerate(final_mol_list):
                mol_graph_simple = []
                for mol_id in range(len(mol)):
                    mol_graph_simple.extend([mol_to_graph_data(mol[mol_id])])
                graph_list.append(mol_graph_simple)
            feature_mols = []
            for i, graphs in enumerate(graph_list):
                _, _, feat_graph = self.embed(graphs)
                feat_bond = self.embed.bond_info(final_smiles_list[i], bond_idx_list[i])
                try:
                    feature_mol = self.action_binding[0](feat_graph.to(self.device), feat_bond.to(self.device))
                except:
                    print('error')
                feature_mols.append(feature_mol)
            ac_seconds = []
            ac_second_probs = []
            log_ac_second_probs = []
            for i, feat_mol in enumerate(feature_mols):
                logits_second = self.action_binding[1](feat_mol)
                ac_second_prob = torch.softmax(logits_second, dim=0).transpose(0, 1) + 1e-8
                log_ac_second_prob = ac_second_prob.log()
                
                ac_second_hot = self.gumbel_softmax(ac_second_prob, tau=self.tau, hard=True, g_ratio=1e-3)
                ac_second_prob = torch.cat(
                    [ac_second_prob, ac_second_prob.new_zeros(1, self.max_action - ac_second_prob.size(1))],
                    -1)
                log_ac_second_prob = torch.cat(
                    [log_ac_second_prob, log_ac_second_prob.new_zeros(1, self.max_action - log_ac_second_prob.size(1))],
                    -1)
                
                ac_second = torch.argmax(ac_second_hot, dim=-1)
                ac_seconds.append(ac_second)
                ac_second_probs.append(ac_second_prob)
                log_ac_second_probs.append(log_ac_second_prob)
            ac_seconds = torch.cat(ac_seconds, dim=0)
            log_ac_second_probs = torch.cat(log_ac_second_probs, dim=0)
            ac_second_probs = torch.cat(ac_second_probs, dim=0)
            
            ac_prob = torch.cat([ac_motif_prob, ac_second_probs], dim=1)
            log_ac_prob = torch.cat([log_ac_motif_prob, log_ac_second_probs], dim=1)
            ac = torch.stack([ac_first, ac_seconds], dim=1)
        else:
            graph_emb_motif = graph_emb.repeat(self.motif_type_num, 1)
            att_emb_motif = self.action_motif[0](graph_emb_motif, cand_graph_emb) + self.action_motif[1](
                graph_emb_motif) \
                            + self.action_motif[2](cand_graph_emb)
            logits_motif = self.action_motif[3](att_emb_motif)
            ac_motif_prob = torch.softmax(logits_motif, dim=0).transpose(0, 1) + 1e-8

            # How to propgate through the network?
            log_ac_motif_prob = ac_motif_prob.log()
            ac_motif_hot = self.gumbel_softmax(ac_motif_prob, tau=self.tau, hard=True, dim=1)
            ac_first = torch.argmax(ac_motif_hot, dim=-1)

            fragment_smi = self.cand[ac_first]['smi']
            starting_mol_list = []
            for x in o:
                starting_mol_list.append(Chem.MolFromSmiles(x['smi']))
            fragment_mol_list = []
            fragment_mol_list.append(Chem.MolFromSmiles(fragment_smi))
            start_idx_list = []
            for mol in starting_mol_list:
                start_idx_list.append(get_att_points(mol))
            end_idx_list = []
            for mol in fragment_mol_list:
                end_idx_list.append(get_att_points(mol))

            final_mol_list = []
            final_smiles_list = []
            bond_idx_list = []
            for i, starting_mol in enumerate(starting_mol_list):
                _mol = []
                mol_tmp = []
                smiles_list = []
                bond_idx = []
                for idx_begin in range(len(start_idx_list[i])):
                    starting_mol_local = copy.deepcopy(starting_mol)
                    neighbors = starting_mol_local.GetAtomWithIdx(start_idx_list[i][idx_begin]).GetNeighbors()
                    for neighbor in neighbors:
                        neighbor.SetProp("Neighbors", "possible_binding_site")
                    starting_mol_local = Chem.RWMol(starting_mol_local)
                    starting_mol_local.RemoveAtom(start_idx_list[i][idx_begin])

                    for idx_end in range(len(end_idx_list[i])):
                        fragment_mol_local = copy.deepcopy(fragment_mol_list[i])
                        frag_neighbors = fragment_mol_local.GetAtomWithIdx(end_idx_list[i][idx_end]).GetNeighbors()
                        for frag_neighbor in frag_neighbors:
                            frag_neighbor.SetProp("Neighbors", "possible_binding_site")
                        fragment_mol_local = Chem.RWMol(fragment_mol_local)
                        fragment_mol_local.RemoveAtom(end_idx_list[i][idx_end])
                        comb_mol = Chem.CombineMols(starting_mol_local, fragment_mol_local)
                        idx = []
                        for atom in comb_mol.GetAtoms():
                            try:
                                if atom.GetProp('Neighbors') == 'possible_binding_site':
                                    idx.append(atom.GetIdx())
                            except:
                                continue
                        ed_mol = Chem.EditableMol(comb_mol)
                        ed_mol.AddBond(idx[0], idx[1], order=Chem.rdchem.BondType.SINGLE)
                        idx = []
                        mol_single = ed_mol.GetMol()
                        for atom in mol_single.GetAtoms():
                            atom.SetAtomMapNum(atom.GetIdx())
                        for atom in mol_single.GetAtoms():
                            try:
                                if atom.GetProp('Neighbors') == 'possible_binding_site':
                                    idx.append(atom.GetAtomMapNum())
                            except:
                                continue
                        mol_smi = Chem.MolToSmiles(mol_single)
                        smiles_list.extend([mol_smi])
                        mol_single = Chem.MolFromSmiles(mol_smi)
                        idx_set = set(idx)
                        for bond in mol_single.GetBonds():
                            if bond.GetBeginAtom().GetAtomMapNum() in idx_set and bond.GetEndAtom().GetAtomMapNum() in idx_set:
                                bond_idx.extend([bond.GetIdx()])
                        mol_tmp.extend([mol_single])
                bond_idx_list.append(bond_idx)
                final_smiles_list.append(smiles_list)
                final_mol_list.append(mol_tmp)
            graph_list = []
            for i, mol in enumerate(final_mol_list):
                mol_graph_simple = []
                for mol_id in range(len(mol)):
                    mol_graph_simple.extend([mol_to_graph_data(mol[mol_id])])
                graph_list.append(mol_graph_simple)
            for i, graphs in enumerate(graph_list):
                _, _, feat_graph = self.embed.forward(graphs)
                feat_bond = self.embed.bond_info(final_smiles_list[i], bond_idx_list[i])
                feature_mol = self.action_binding[0](feat_graph.to(self.device), feat_bond.to(self.device))
            logits_second = self.action_binding[1](feature_mol)
            # logits_second = self.action_binding[1](emb_first_batch)
            ac_second_prob = torch.softmax(logits_second, dim=0).transpose(0, 1) + 1e-8
            log_ac_second_prob = ac_second_prob.log()  # torch.Size([N, 1])
            ac_second_hot = self.gumbel_softmax(ac_second_prob, tau=self.tau, hard=True, g_ratio=1e-3)
            ac_second = torch.argmax(ac_second_hot, dim=-1)
            ac_second_prob = torch.cat([ac_second_prob, ac_second_prob.new_zeros(
                1, self.max_action - ac_second_prob.size(1))], -1, )
            log_ac_second_prob = torch.cat([log_ac_second_prob, log_ac_second_prob.new_zeros(
                1, self.max_action - log_ac_second_prob.size(1))], -1)
            ac_prob = torch.cat([ac_motif_prob, ac_second_prob], dim=1)
            log_ac_prob = torch.cat([log_ac_motif_prob, log_ac_second_prob], dim=1)
            ac = torch.stack([ac_first, ac_second], dim=1)
        return ac, ac_prob, log_ac_prob, final_smiles_list, bond_idx_list

    def _distribution(self, ac_prob):

        ac_prob_split = torch.split(ac_prob, [len(FRAG_VOCAB), self.max_action], dim=1)
        dists = [Categorical(probs=pr) for pr in ac_prob_split]
        return dists

    def _log_prob_from_distribution(self, dists, act):
        log_probs = [p.log_prob(act[:, i]) for i, p in enumerate(dists)]
        return torch.stack(log_probs, dim=1)
