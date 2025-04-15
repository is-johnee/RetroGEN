import os
import torch
import pickle
import pandas as pd
import numpy as np
import networkx as nx
import random
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
from itertools import repeat
# from mol_features import atom_to_feature_vector, bond_to_feature_vector
from reactive_site_features import Local_FromSmiles
from tqdm import tqdm, trange

# from dataset_attentivefp import atom_attr, bond_attr, MolDataset

# num_bond_type = 6
# num_atom_type = 119
# allowable node and edge features
num_atom_type = 120 #including the extra mask tokens=119
num_chirality_tag = 3 # original =3. including the extra mask tokens=3
num_bond_type = 6 #including aromatic and self-loop edge, and extra masked tokens
num_bond_direction = 3 # original =3, inlcuding the extra mask tokens=3

allowable_features = {
    'possible_atomic_num_list' : list(range(0, 119)),
    'possible_formal_charge_list' : [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
    'possible_chirality_list' : [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER
    ],
    'possible_hybridization_list' : [
        Chem.rdchem.HybridizationType.S,
        Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2, Chem.rdchem.HybridizationType.UNSPECIFIED
    ],
    'possible_numH_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8],
    'possible_implicit_valence_list' : [0, 1, 2, 3, 4, 5, 6],
    'possible_degree_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'possible_bonds' : [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC
    ],
    'possible_bond_dirs' : [ # only for double bond stereo information
        Chem.rdchem.BondDir.NONE,
        Chem.rdchem.BondDir.ENDUPRIGHT,
        Chem.rdchem.BondDir.ENDDOWNRIGHT
    ]
}
# allowable_features = {
#     'possible_atomic_num_list': list(range(1, 119)) + ['misc'],
#     'possible_formal_charge_list': [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 'misc'],
#     'possible_chirality_list': [
#         Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
#         Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
#         Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
#         Chem.rdchem.ChiralType.CHI_OTHER,
#         'misc'
#     ],
#     'possible_hybridization_list': [
#         Chem.rdchem.HybridizationType.S,
#         Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
#         Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
#         Chem.rdchem.HybridizationType.SP3D2, Chem.rdchem.HybridizationType.UNSPECIFIED, 'misc'
#     ],
#     'possible_numH_list': [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
#     'possible_number_radical_e_list': [0, 1, 2, 3, 4, 'misc'],
#     'possible_implicit_valence_list': [0, 1, 2, 3, 4, 5, 6, 'misc'],
#     'possible_degree_list': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'misc'],
#     'possible_is_aromatic_list': [False, True, 'misc'],
#     'possible_is_in_ring_list': [False, True, 'misc'],
#     'possible_bonds': [
#         Chem.rdchem.BondType.SINGLE,
#         Chem.rdchem.BondType.DOUBLE,
#         Chem.rdchem.BondType.TRIPLE,
#         Chem.rdchem.BondType.AROMATIC,
#         'misc'
#     ],
#     'possible_bond_dirs': [  # only for double bond stereo information
#         Chem.rdchem.BondDir.NONE,
#         Chem.rdchem.BondDir.ENDUPRIGHT,
#         Chem.rdchem.BondDir.ENDDOWNRIGHT,
#         'misc'
#     ],
#     'possible_bond_stereo_list': [
#         Chem.rdchem.BondStereo.STEREONONE,
#         Chem.rdchem.BondStereo.STEREOZ,
#         Chem.rdchem.BondStereo.STEREOE,
#         Chem.rdchem.BondStereo.STEREOCIS,
#         Chem.rdchem.BondStereo.STEREOTRANS,
#         Chem.rdchem.BondStereo.STEREOANY,
#         'misc'
#     ],
#     'possible_is_conjugated_list': [False, True, 'misc'],
# }

def mol_to_graph_data(mol):
    """
    Converts rdkit mol object to graph Data object required by the pytorch
    geometric package. NB: Uses simplified atom and bond features, and represent
    as indices
    :param mol: rdkit mol object
    :return: graph data object with the attributes: x, edge_index, edge_attr
    """
    # atoms
    num_atom_features = 2   # atom type,  chirality tag
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_feature = [allowable_features['possible_atomic_num_list'].index(atom.GetAtomicNum())] + \
                       [allowable_features['possible_chirality_list'].index(atom.GetChiralTag())]
        atom_features_list.append(atom_feature)
    x = torch.tensor(np.array(atom_features_list), dtype=torch.long)

    # bonds
    num_bond_features = 2   # bond type, bond direction
    if len(mol.GetBonds()) > 0: # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_feature = [allowable_features['possible_bonds'].index(
                bond.GetBondType())] + [allowable_features[
                                            'possible_bond_dirs'].index(
                bond.GetBondDir())]
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = torch.tensor(np.array(edge_features_list),
                                 dtype=torch.long)
    else:   # mol has no bonds
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    return data

def mol_to_graph_data_obj_simple(mol):
    """
    Converts rdkit mol object to graph Data object required by the pytorch
    geometric package. NB: Uses simplified atom and bond features, and represent
    as indices
    :param mol: rdkit mol object
    :return: graph data object with the attributes: x, edge_index, edge_attr
    """
    num_atom_features = 9  # atom type,  chirality tag
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_feature = [allowable_features['possible_atomic_num_list'].index(atom.GetAtomicNum())] + \
                       [allowable_features['possible_chirality_list'].index(atom.GetChiralTag())] + \
                       [allowable_features['possible_degree_list'].index(atom.GetDegree())] + \
                       [allowable_features['possible_formal_charge_list'].index(atom.GetFormalCharge())] + \
                       [allowable_features['possible_numH_list'].index(atom.GetTotalNumHs())] + \
                       [allowable_features['possible_number_radical_e_list'].index(atom.GetNumRadicalElectrons())] + \
                       [allowable_features['possible_hybridization_list'].index(atom.GetHybridization())] + \
                       [allowable_features['possible_is_aromatic_list'].index(atom.GetIsAromatic())] + \
                       [allowable_features['possible_is_in_ring_list'].index(atom.IsInRing())]
        atom_features_list.append(atom_feature)
    x = torch.tensor(np.array(atom_features_list), dtype=torch.long)

    # bonds
    num_bond_features = 4  # bond type, bond direction
    if len(mol.GetBonds()) > 0:  # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_feature = [allowable_features['possible_bonds'].index(bond.GetBondType())] + \
                           [allowable_features['possible_bond_dirs'].index(bond.GetBondDir())] + \
                           [allowable_features['possible_bond_stereo_list'].index(bond.GetStereo())] + \
                           [allowable_features['possible_is_conjugated_list'].index(bond.GetIsConjugated())]
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = torch.tensor(np.array(edge_features_list),
                                 dtype=torch.long)
    else:  # mol has no bonds
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    return data


def mol_to_fp_data(mol):
    rdkbi = {}
    num_atom_features = 9  # atom type,  chirality tag
    atom_features_list = []
    # atom_features_list.extend(np.array(AllChem.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=1024),dtype=np.int8))
    # atom_features_list.extend(np.array(Chem.RDKFingerprint(mol, maxPath=5, bitInfo=rdkbi), dtype=np.int8))
    atom_features_list.extend(np.array(MACCSkeys.GenMACCSKeys(mol), dtype=np.int8))
    x = torch.tensor(np.array(atom_features_list), dtype=torch.long)
    data = Data(x=x)
    return data


def drop_nodes(data, aug_ratio):
    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    drop_num = int(node_num * aug_ratio)

    idx_drop = np.random.choice(node_num, drop_num, replace=False)
    idx_nondrop = [n for n in range(node_num) if not n in idx_drop]
    # print('idx_drop: ', idx_drop)
    # print('idx_nondrop sorted: ', idx_nondrop)
    idx_dict = {idx_nondrop[n]: n for n in list(range(len(idx_nondrop)))}
    # print(idx_dict)

    edge_index = data.edge_index.numpy()
    idx_drop = set(idx_drop)
    idx_nondrop_edge = []
    for i in range(edge_index.shape[1]):  # 边的数目 无向图，边是双向的
        tmp = set(edge_index[:, i])
        if not idx_drop.intersection(tmp):
            idx_nondrop_edge.append(i)
            edge_index[0, i] = idx_dict[edge_index[0, i]]
            edge_index[1, i] = idx_dict[edge_index[1, i]]

    edge_index = torch.from_numpy(edge_index[:, idx_nondrop_edge])
    edge_attr = data.edge_attr[idx_nondrop_edge, :]

    try:
        data.x = data.x[idx_nondrop]
        data.edge_index = edge_index
        data.edge_attr = edge_attr
    except:
        data = data
    return data


def drop_nodes_random(data, aug_ratio):
    # print('data.x:', data.x)
    # print('data.edge_index:', data.edge_index)
    # print('data.edge_attr:', data.edge_attr)
    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    drop_num = int(node_num * aug_ratio)

    idx_drop = np.random.choice(node_num, drop_num, replace=False)
    idx_nondrop = [n for n in range(node_num) if not n in idx_drop]
    # print('idx_drop,', idx_drop)
    # print('idx_nondrop', idx_nondrop)

    # drop node features
    ## data.x = data.x[idx_nondrop]

    # modify edge index and feature
    edge_index = data.edge_index.numpy()
    idx_drop = set(idx_drop)
    idx_nondrop_edge = []
    for i in range(edge_index.shape[1]):
        tmp = set(edge_index[:, i])
        if not idx_drop.intersection(tmp):
            idx_nondrop_edge.append(i)

    edge_index = torch.from_numpy(edge_index[:, idx_nondrop_edge])
    edge_attr = data.edge_attr[idx_nondrop_edge, :]

    data.edge_index = edge_index
    data.edge_attr = edge_attr
    if data.edge_index.shape[1] != data.edge_attr.shape[0]:
        print('data dropping failed!')
        return
    # print(data.x)
    # print(data.edge_index)
    # print(data.edge_attr)

    return data


def drop_nodes_nonC(data, aug_ratio):
    # print('data.x:', data.x)
    # print('data.edge_index:', data.edge_index)
    # print('data.edge_attr:', data.edge_attr)
    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    drop_num = int(node_num * aug_ratio)

    atom_number = data.x[:, 0].numpy()
    c_idx = np.where(atom_number == 5)[0]
    nonc_idx = np.where(atom_number != 5)[0]

    # get drop_idx
    if drop_num <= len(nonc_idx):
        idx_drop = np.random.choice(nonc_idx, drop_num, replace=False)
    else:
        tmp = np.random.choice(c_idx, drop_num - len(nonc_idx), replace=False)
        idx_drop = np.concatenate([nonc_idx, tmp], axis=0)

    idx_nondrop = [n for n in range(node_num) if not n in idx_drop]

    # print('idx_drop,', idx_drop)
    # print('idx_nondrop', idx_nondrop)

    # drop node features
    ## data.x = data.x[idx_nondrop]

    # modify edge index and feature
    edge_index = data.edge_index.numpy()
    idx_drop = set(idx_drop)
    idx_nondrop_edge = []
    for i in range(edge_index.shape[1]):
        tmp = set(edge_index[:, i])
        if not idx_drop.intersection(tmp):
            idx_nondrop_edge.append(i)

    edge_index = torch.from_numpy(edge_index[:, idx_nondrop_edge])
    edge_attr = data.edge_attr[idx_nondrop_edge, :]

    data.edge_index = edge_index
    data.edge_attr = edge_attr
    if data.edge_index.shape[1] != data.edge_attr.shape[0]:
        print('data dropping failed!')
        return
    # print(data.x)
    # print(data.edge_index)
    # print(data.edge_attr)

    return data


def drop_nodes_C(data, aug_ratio):
    # print('data.x:', data.x)
    # print('data.edge_index:', data.edge_index)
    # print('data.edge_attr:', data.edge_attr)
    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    # 此处应当只能是碳的数目才对

    atom_number = data.x[:, 0].numpy()
    c_idx = np.where(atom_number == 5)[0]
    drop_num = int(len(c_idx) * aug_ratio)  # 此处应当只能是碳的数目才对
    # get drop_idx
    idx_drop = np.random.choice(c_idx, drop_num, replace=False)

    idx_nondrop = [n for n in range(node_num) if not n in idx_drop]

    # print('idx_drop,', idx_drop)
    # print('idx_nondrop', idx_nondrop)

    # drop node features
    ## data.x = data.x[idx_nondrop]

    # modify edge index and feature
    edge_index = data.edge_index.numpy()
    idx_drop = set(idx_drop)
    idx_nondrop_edge = []
    for i in range(edge_index.shape[1]):
        tmp = set(edge_index[:, i])
        if not idx_drop.intersection(tmp):
            idx_nondrop_edge.append(i)

    edge_index = torch.from_numpy(edge_index[:, idx_nondrop_edge])
    edge_attr = data.edge_attr[idx_nondrop_edge, :]

    data.edge_index = edge_index
    data.edge_attr = edge_attr
    if data.edge_index.shape[1] != data.edge_attr.shape[0]:
        print('data dropping failed!')
        return
    # print(data.x)
    # print(data.edge_index)
    # print(data.edge_attr)

    return data


def add_edges(data, aug_ratio):
    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    permute_num = int(edge_num * aug_ratio)

    edge_index = data.edge_index.transpose(0, 1).numpy()
    edge_index_add = np.random.choice(node_num, (permute_num, 2))

    idx_drop = np.random.choice(edge_num, permute_num, replace=False)
    edge_index[idx_drop] = edge_index_add

    data.edge_index = torch.tensor(edge_index).transpose_(0, 1)
    return data


def permute_edges(data, aug_ratio):
    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    permute_num = int(edge_num * aug_ratio)

    edge_index = data.edge_index.transpose(0, 1).numpy()
    edge_index_add = np.random.choice(node_num, (permute_num, 2))

    idx_drop = np.random.choice(edge_num, permute_num, replace=False)
    edge_index[idx_drop] = edge_index_add

    data.edge_index = torch.tensor(edge_index).transpose_(0, 1)
    return data


def drop_edges(data, aug_ratio):
    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    permute_num = int(edge_num * aug_ratio)

    edge_index = data.edge_index.transpose(0, 1).numpy()

    idx_nondrop = np.random.choice(edge_num, edge_num - permute_num, replace=False)
    edge_index = edge_index[idx_nondrop, :]

    data.edge_index = torch.tensor(edge_index).transpose_(0, 1)
    data.edge_attr = data.edge_attr[idx_nondrop, :]
    return data


def subgraph(data, aug_ratio):
    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    sub_num = int(node_num * aug_ratio)

    edge_index = data.edge_index.numpy()

    idx_sub = [np.random.randint(node_num, size=1)[0]]
    # print('idx_sub: ', idx_sub)
    idx_neigh = set([n for n in edge_index[1][edge_index[0] == idx_sub[0]]])
    # print('idx_neigh: ', idx_neigh)

    count = 0
    while len(idx_sub) <= sub_num:
        count = count + 1
        if count > node_num:
            break
        if len(idx_neigh) == 0:
            break
        sample_node = np.random.choice(list(idx_neigh))
        if sample_node in idx_sub:
            continue
        idx_sub.append(sample_node)
        idx_neigh.union(set([n for n in edge_index[1][edge_index[0] == idx_sub[-1]]]))
    # print('idx_sub_final: ', idx_sub)
    # print('idx_neigh_final: ', idx_neigh)

    idx_drop = [n for n in range(node_num) if not n in idx_sub]
    # print('idx_drop: ', idx_drop)
    idx_nondrop = sorted(idx_sub)
    # print('idx_nondrop sorted: ', idx_nondrop)
    idx_dict = {idx_nondrop[n]: n for n in list(range(len(idx_nondrop)))}
    # print(idx_dict)
    edge_index = data.edge_index.numpy()
    idx_drop = set(idx_drop)
    idx_nondrop_edge = []
    for i in range(edge_index.shape[1]):
        tmp = set(edge_index[:, i])
        if not idx_drop.intersection(tmp):
            idx_nondrop_edge.append(i)
            edge_index[0, i] = idx_dict[edge_index[0, i]]
            edge_index[1, i] = idx_dict[edge_index[1, i]]

    edge_index = torch.from_numpy(edge_index[:, idx_nondrop_edge])
    edge_attr = data.edge_attr[idx_nondrop_edge, :]

    try:
        data.x = data.x[idx_nondrop]
        data.edge_index = edge_index
        data.edge_attr = edge_attr
    except:
        data = data
    # print('data.x new', data.x)
    # print('data.x try index')
    # for i in range(data.x.size(0)):
    #     print(i, data.x[i])
    # print('edge_index new', data.edge_index)
    # print('edge_attr new', data.edge_attr)
    return data


def mask_attributes(data, aug_ratio, mask_edge = False):

    node_num, feat_dim = data.x.size()
    mask_num = int(node_num * aug_ratio)

    idx_mask = np.random.choice(node_num, mask_num, replace=False)
    data.x[idx_mask] = torch.tensor([num_atom_type-1, 0])

    if mask_edge:
        edge_index = data.edge_index.numpy()
        idx_mask = set(idx_mask)
        idx_mask_edge = []
        for i in range(edge_index.shape[1]):
            tmp = set(edge_index[:, i])
            if idx_mask.intersection(tmp):
                idx_mask_edge.append(i)
        data.edge_attr[idx_mask_edge] = torch.tensor([num_bond_type-1, 0])

    return data


# 此处也需要修改
def bonds_aug(data, aug_ratio, smiles):
    # 首先判断其是有几个对应的新生成的键，如果多于两个的话，则对边数据进行增强
    # 如果选择了bonds_aug，但是在选择之前是不确定能否进行增强的
    # 这样的话就在这边进行判断
    # 扩充得到的都是正例，同一个batch内的其他的负样本都是负例
    if ((data['bonds_new'].size(0) + data['bonds_changed'].size(0)) > 1):
        # 如果多于一个的话，则对其选择使用bonds_aug  不需要掩蔽键或者官能团等 图的信息不变，改变边的表示
        len_bonds = data['bonds_new'].size(0) + data['bonds_changed'].size(0)
        bonds_changed = data['bonds_changed']
        bonds_new = data['bonds_new']
        bonds = torch.cat((bonds_new, bonds_changed))
        # 获取这两个边对应的idx
        changed = random.randint(0, len_bonds - 1)
        bd = bonds[changed]
        bg_atom_idx, ed_atom_idx = getbondwithatm(smiles, bd)
        # feat = Local_FromSmiles(smiles=smiles, index=bg_atom_idx, radius=2)
    else:
        # 选择其他增强方式，但是对于那些没有新增的边的情况如何处理，随机（调用随机函数，随机选择上面的一种）选择其他的
        data = drop_edges(data, aug_ratio)
    return data


def getbondwithatm(smiles, atomidx):
    # 注意此处识别到的atom_idx指的并非是从头开始得到的
    mol = Chem.MolFromSmiles(smiles)
    bond_idx = atomidx[2]
    bond = mol.GetBondWithIdx(bond_idx)
    # 并且获取到对应的特征如何求？
    return bond.GetBeginAtom().GetAtomIdx(), bond.GetEndAtom().GetAtomIdx()


def randrop(data, aug_ratio, mask_edge=False):
    ri = np.random.randint(10)
    # print('ri={}'.format(ri))
    if ri == 0:
        data = drop_nodes_random(data, aug_ratio)
    elif ri == 1:
        data = drop_nodes_nonC(data, aug_ratio)
    elif ri == 2:
        data = data
    elif ri == 3:
        data = drop_edges(data, aug_ratio)
    elif ri == 4:
        data = permute_edges(data, aug_ratio)
    elif ri == 5:
        data = mask_attributes(data, aug_ratio)
    elif ri == 6:
        data = subgraph(data, aug_ratio)
    elif ri == 7:
        data = drop_nodes(data, aug_ratio)
    elif ri == 8:
        # try:
        data = drop_nodes_C(data, aug_ratio)
        # except:
        #     data = randrop(data, aug_ratio)
    elif ri == 9:
        data = add_edges(data, aug_ratio)
    return data


class MoleculeDataset(InMemoryDataset):
    def __init__(self,
                 root,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 dataset=None,
                 aug='none', aug_ratio=None):
        self.dataset = dataset
        self.aug = aug
        self.aug_ratio = aug_ratio
        self.get_count = 0
        # self.raw = os.path.join(root, self.dataset, 'raw')
        # self.processed = os.path.join(root, self.dataset, 'processed')
        self.transform, self.pre_transform, self.pre_filter = transform, pre_transform, pre_filter
        super(MoleculeDataset, self).__init__(root, transform, pre_transform,
                                              pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        # raw_file = pd.read_csv('dataset/raw/USPTO/data/uspto_full_new.csv', sep=',')
        # self.original_smiles = raw_file['products'].tolist()

    def get(self, idx):
        # 获取单个分子
        # 应该优先执行对键进行增强
        data = Data()
        # 如果此处是对应的下游测试任务，则需要指定label y
        for key in self.data.keys:
            if key != 'product' and key != 'bonds' and key != 'num_bonds':
                item, slices = self.data[key], self.slices[key]  # 对应的所有的数据
                s = list(repeat(slice(None), item.dim()))  # item.dim()获取的是item的维度
                #            print(key, item) 这里要进行修改为__cat_dim__ 每一个对应的dim不同
                s[data.__cat_dim__(key, item)] = slice(slices[idx], slices[
                    idx + 1])  # __ cat_dim __返回值表示相应矩阵拼接的维度。按行或列拼接，一般用于节点或者结果矩阵的拼接
                # https://www.i4k.xyz/article/qq_37252519/119357519 +1指的是间隔的下标值
                # slice切片对象，用以获取切片
                data[key] = item[s]  # http://www.4k8k.xyz/article/qq_41795143/114281387
            else:
                item, slices = self.data[key], self.slices[key]
                data[key] = item[idx]
        num_bonds_changes = data.num_bonds.item()
        if num_bonds_changes == 0:
            pass
        elif num_bonds_changes == 1:
            # 对图进行增强 边仅使用这一个 无法对边进行增强
            data = randrop(data, self.aug_ratio)
            data.bond_aug = data.bonds[0]
        else:  # !=1 为0排除
            # 随机对边和图进行增强——根据数据集的比例来看，为1的占很多，因此这里最好全部为对边进行增强
            id = np.random.choice([1])
            # id = np.random.choice([0, 1])
            if id == 0:
                # 对图进行增强
                # 但是输入的内容是有两个的，如果对应的bond==0如何处理？这样也不会保证两次都是使用同一种增强方式？
                data = randrop(data, self.aug_ratio)
                data.bond_aug = data.bonds[0]  # 应当是随机选择一个键，两种情况下都是选择相同的键，如果使用随机选择，则不能保证下一个也是选择这个
            else:
                # 对边进行增强 随机选择一个边 作为对应的边的信息 新增一个信息作为其对应的键？
                bonds_list_odd = list(i for i in range(len(data.bonds)) if i % 2 != 0)
                bonds_list_even = list(i for i in range(len(data.bonds)) if i % 2 == 0)
                # bonds_list_odd = list(data.bonds[i][2] for i in range(len(data.bonds)) if i % 2 != 0)
                # bonds_list_even = list(data.bonds[i][2] for i in range(len(data.bonds)) if i % 2 == 0)
                # 如果为0，则在所有的偶数里进行选择 如果为1，则在所有的奇数里进行选择 这样能够保证其不会重复
                bond_idx = 0
                if self.bond_idx == 0:
                    try:
                        bond_idx = np.random.choice(bonds_list_even)
                    except:
                        print('empty!')
                else:
                    bond_idx = np.random.choice(bonds_list_odd)
                data.bond_aug = data.bonds[bond_idx]
        # if self.aug_ratio == 0:
        #     pass
        # else:
        #     data = randrop(data, aug_ratio=self.aug_ratio)
        return data

    @property
    def raw_file_names(self):
        # file_name_list = os.listdir(self.raw)
        # assert len(file_name_list) == 1     # currently assume we have a
        # # single raw file
        return ['{}.pickle'.format(self.dataset)]  # raw_paths调用此

    # 调用父类初始化时会优先执行@property函数
    @property
    def processed_file_names(self):  # 返回一个处理后的文件列表，包含processed_dir中的文件目录，根据列表内容决定哪些数据已经处理过从而可以跳过处理；
        # return 'geometric_data_full_new_processed.pt'  # 预处理之后保存到此处 好像是先执行的此处，之后才执行process这个函数
        # 如果没找到这个文件则会执行process获得处理后的文件
        return ['{}_reset_mols.pt'.format(self.dataset)]

    def download(self):
        pass

    # 修改版本，改动使得其能够满足在self.collate()之后得到结果
    def process(self):
        # 修改模型为只要是改变的都作为reaction center
        # https://www.cxymm.net/article/qq_32583189/110196193
        data_list = []
        # atom_mapping, reactants, products, bonds_new, bonds_changed, bonds_lost= _load_uspto_dataset(self.raw_paths[0])#注意这里如果文件名不进行更改的话是有问题的
        reaction = open(self.raw_paths[0], 'rb')
        # 这里不能只是这一个数据，不然不管是什么任务都是这个
        reaction_data = pickle.load(reaction)
        # num_bonds_changed = reaction_data['num_changed']
        # bonds_changed = reaction_data['bonds_changed']
        reaction_data = reaction_data[reaction_data['num_changed'] >= 1]
        # 保存
        # new_reaction_data.to_csv('dataset/raw/USPTO/data/new_uspto_mit.csv')
        products = reaction_data['products'].reset_index(drop=True)
        bonds_changed = reaction_data['bonds_changes'].reset_index(drop=True)
        num_bonds_changed = reaction_data['num_changed'].reset_index(drop=True)
        for i in trange(len(products)):
            try:
                rdkit_mol = Chem.MolFromSmiles(products[i])
                data = mol_to_graph_data(rdkit_mol)
                data.num_bonds = torch.tensor(num_bonds_changed[i])
                data.bonds = bonds_changed[i]
                data.product = products[i]
                data_list.append(data)
                # data_smiles_list.append(products[i])
            except:
                print('Exceptions while processing molecule:{}'.format(products[i]))

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        data, slices = self.collate(data_list)  # 注意将这里的数据转换为一定的格式，不然无法用这个进行处理
        torch.save((data, slices), self.processed_paths[0])  # 保存geometric_data_processed.pt文件


class USPTODataset(InMemoryDataset):
    def __init__(self,
                 root,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 dataset=None,
                 aug='none', aug_ratio=None):
        """
        Adapted from qm9.py. Disabled the download functionality
        :param root: directory of the dataset, containing a raw and processed
        dir. The raw dir should contain the file containing the smiles, and the
        processed dir can either empty or a previously processed file
        :param dataset: name of the dataset. Currently only implemented for
        zinc250k, chembl_with_labels, tox21, hiv, bace, bbbp, clintox, esol,
        freesolv, lipophilicity, muv, pcba, sider, toxcast
        :param empty: if True, then will not load any data obj. For
        initializing empty dataset
        """
        # 调用父类进行初始化：计算其对应的有继承的方法
        # 初始化的时候读入这个文件，之后再进行处理？
        self.dataset = dataset
        # self.root = root
        self.aug = aug
        self.aug_ratio = aug_ratio
        # self.bond_idx = bond_idx
        self.get_count = 0
        # self.raw = os.path.join(root, self.dataset, 'raw')
        # self.processed = os.path.join(root, self.dataset, 'processed')
        self.transform, self.pre_transform, self.pre_filter = transform, pre_transform, pre_filter
        super(USPTODataset, self).__init__(root, transform, pre_transform,
                                           pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        # raw_file = pd.read_csv('dataset/raw/USPTO/data/uspto_full_new.csv', sep=',')
        # self.original_smiles = raw_file['products'].tolist()

    def get(self, idx):
        # 获取单个分子
        # 应该优先执行对键进行增强
        data = Data()
        # 如果此处是对应的下游测试任务，则需要指定label y
        for key in self.data.keys:
            if key != 'product' and key != 'bonds' and key != 'num_bonds':
                item, slices = self.data[key], self.slices[key]  # 对应的所有的数据
                s = list(repeat(slice(None), item.dim()))  # item.dim()获取的是item的维度
                #            print(key, item) 这里要进行修改为__cat_dim__ 每一个对应的dim不同
                s[data.__cat_dim__(key, item)] = slice(slices[idx], slices[
                    idx + 1])  # __ cat_dim __返回值表示相应矩阵拼接的维度。按行或列拼接，一般用于节点或者结果矩阵的拼接
                # https://www.i4k.xyz/article/qq_37252519/119357519 +1指的是间隔的下标值
                # slice切片对象，用以获取切片
                data[key] = item[s]  # http://www.4k8k.xyz/article/qq_41795143/114281387
            else:
                item, slices = self.data[key], self.slices[key]
                data[key] = item[idx]
            # 之前的策略是如果多于一个才进行键的扩增当作正例
            # if data.bonds[0] != -1 and data.bonds[1] != -1:
            #     # 说明是有两条边的情况 判断其应该用哪个进行扩增
            #     if (self.bond_idx == 0):
            #         data.aug_bond = 0
            #     else:
            #         data.aug_bond = 1
            # elif (data.bonds[0] != -1):  # 设计为随机选择一个
            #     data = randrop(data, self.aug_ratio)
            #     data.aug_bond = 0
            # else:  # data.bonds[1]!=-1
            #     data = randrop(data, self.aug_ratio)
            #     data.aug_bond = 1

            # 对数据进行处理 因为是两组数据 对应两个正例
            # 如果分子可能断开的键的数目>=2个，则随机选取两个 一个作为前面的dataset 一个作为后面的dataset
            # 如果只有一个键
        # num_bonds_changes = data.num_bonds.item()
        # if num_bonds_changes == 0:
        #     pass
        # elif num_bonds_changes == 1:
        #     # 对图进行增强 边仅使用这一个 无法对边进行增强
        #     data = randrop(data, self.aug_ratio)
        #     data.bond_aug = data.bonds[0]
        # else:  # !=1 为0排除
        #     # 随机对边和图进行增强——根据数据集的比例来看，为1的占很多，因此这里最好全部为对边进行增强
        #     id = np.random.choice([1])
        #     # id = np.random.choice([0, 1])
        #     if id == 0:
        #         # 对图进行增强
        #         # 但是输入的内容是有两个的，如果对应的bond==0如何处理？这样也不会保证两次都是使用同一种增强方式？
        #         data = randrop(data, self.aug_ratio)
        #         data.bond_aug = data.bonds[0]  # 应当是随机选择一个键，两种情况下都是选择相同的键，如果使用随机选择，则不能保证下一个也是选择这个
        #     else:
        #         # 对边进行增强 随机选择一个边 作为对应的边的信息 新增一个信息作为其对应的键？
        #         bonds_list_odd = list(i for i in range(len(data.bonds)) if i % 2 != 0)
        #         bonds_list_even = list(i for i in range(len(data.bonds)) if i % 2 == 0)
        #         # 如果为0，则在所有的偶数里进行选择 如果为1，则在所有的奇数里进行选择 这样能够保证其不会重复
        #         bond_idx = 0
        #         if self.bond_idx == 0:
        #             try:
        #                 bond_idx = np.random.choice(bonds_list_even)
        #             except:
        #                 print('empty!')
        #         else:
        #             bond_idx = np.random.choice(bonds_list_odd)
        #         data.bond_aug = data.bonds[bond_idx]
        # if self.aug_ratio == 0:
        #     pass
        # else:
        #     data = randrop(data, aug_ratio=self.aug_ratio)
        return data

    @property
    def raw_file_names(self):
        # file_name_list = os.listdir(self.raw)
        # assert len(file_name_list) == 1     # currently assume we have a
        # # single raw file
        return ['{}.pickle'.format(self.dataset)]  # raw_paths调用此

    # 调用父类初始化时会优先执行@property函数
    @property
    def processed_file_names(self):  # 返回一个处理后的文件列表，包含processed_dir中的文件目录，根据列表内容决定哪些数据已经处理过从而可以跳过处理；
        # return 'geometric_data_full_new_processed.pt'  # 预处理之后保存到此处 好像是先执行的此处，之后才执行process这个函数
        # 如果没找到这个文件则会执行process获得处理后的文件
        return ['{}.pt'.format(self.dataset)]

    def download(self):
        pass

    # 修改版本，改动使得其能够满足在self.collate()之后得到结果
    def process(self):
        # 修改模型为只要是改变的都作为reaction center
        # https://www.cxymm.net/article/qq_32583189/110196193
        data_smiles_list = []
        data_list = []
        # atom_mapping, reactants, products, bonds_new, bonds_changed, bonds_lost= _load_uspto_dataset(self.raw_paths[0])#注意这里如果文件名不进行更改的话是有问题的
        reaction = open(self.raw_paths[0], 'rb')
        # 这里不能只是这一个数据，不然不管是什么任务都是这个
        reaction_data = pickle.load(reaction)
        # num_bonds_changed = reaction_data['num_changed']
        # bonds_changed = reaction_data['bonds_changed']
        # reaction_data = reaction_data[reaction_data['num_changed'] >= 1]
        # 保存
        # new_reaction_data.to_csv('dataset/raw/USPTO/data/new_uspto_mit.csv')
        products = reaction_data['products']
        bonds_changed = reaction_data['bonds_changes']
        num_bonds_changed = reaction_data['num_changed']
        # 对数据进行处理
        num_bonds_changed.mask(num_bonds_changed == 9, 8, inplace=True)
        num_bonds_changed.mask(num_bonds_changed == 10, 9, inplace=True)
        num_bonds_changed.mask(num_bonds_changed == 11, 10, inplace=True)
        num_bonds_changed.mask(num_bonds_changed == 13, 11, inplace=True)
        num_bonds_changed.mask(num_bonds_changed == 15, 12, inplace=True)
        num_bonds_changed.mask(num_bonds_changed == 17, 13, inplace=True)
        for i in trange(len(products)):
            # idx=alter_idx[i]
            try:
                rdkit_mol = Chem.MolFromSmiles(products[i])
                # data = mol2graph(rdkit_mol)
                data = mol_to_graph_data(rdkit_mol)
                data.num_bonds = torch.tensor(num_bonds_changed[i])
                if num_bonds_changed[i] == 0:
                    data.bonds = [(-1, -1, -1)]
                else:
                    data.bonds = bonds_changed[i]
                data_list.append(data)
                data_smiles_list.append(products[i])
            except:
                print('exception')

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        data, slices = self.collate(data_list)  # 注意将这里的数据转换为一定的格式，不然无法用这个进行处理
        torch.save((data, slices), self.processed_paths[0])  # 保存geometric_data_processed.pt文件


class USPTODataset(InMemoryDataset):
    def __init__(self,
                 root,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 dataset=None,
                 aug='none', aug_ratio=None):
        """
        Adapted from qm9.py. Disabled the download functionality
        :param root: directory of the dataset, containing a raw and processed
        dir. The raw dir should contain the file containing the smiles, and the
        processed dir can either empty or a previously processed file
        :param dataset: name of the dataset. Currently only implemented for
        zinc250k, chembl_with_labels, tox21, hiv, bace, bbbp, clintox, esol,
        freesolv, lipophilicity, muv, pcba, sider, toxcast
        :param empty: if True, then will not load any data obj. For
        initializing empty dataset
        """
        # 调用父类进行初始化：计算其对应的有继承的方法
        # 初始化的时候读入这个文件，之后再进行处理？
        self.dataset = dataset
        # self.root = root
        self.aug = aug
        self.aug_ratio = aug_ratio
        # self.bond_idx = bond_idx
        self.get_count = 0
        # self.raw = os.path.join(root, self.dataset, 'raw')
        # self.processed = os.path.join(root, self.dataset, 'processed')
        self.transform, self.pre_transform, self.pre_filter = transform, pre_transform, pre_filter
        super(USPTODataset, self).__init__(root, transform, pre_transform,
                                           pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        # raw_file = pd.read_csv('dataset/raw/USPTO/data/uspto_full_new.csv', sep=',')
        # self.original_smiles = raw_file['products'].tolist()

    def get(self, idx):
        # 获取单个分子
        # 应该优先执行对键进行增强
        data = Data()
        # 如果此处是对应的下游测试任务，则需要指定label y
        for key in self.data.keys:
            if key != 'product' and key != 'bonds' and key != 'num_bonds':
                item, slices = self.data[key], self.slices[key]  # 对应的所有的数据
                s = list(repeat(slice(None), item.dim()))  # item.dim()获取的是item的维度
                #            print(key, item) 这里要进行修改为__cat_dim__ 每一个对应的dim不同
                s[data.__cat_dim__(key, item)] = slice(slices[idx], slices[
                    idx + 1])  # __ cat_dim __返回值表示相应矩阵拼接的维度。按行或列拼接，一般用于节点或者结果矩阵的拼接
                # https://www.i4k.xyz/article/qq_37252519/119357519 +1指的是间隔的下标值
                # slice切片对象，用以获取切片
                data[key] = item[s]  # http://www.4k8k.xyz/article/qq_41795143/114281387
            else:
                item, slices = self.data[key], self.slices[key]
                data[key] = item[idx]
            # 之前的策略是如果多于一个才进行键的扩增当作正例
            # if data.bonds[0] != -1 and data.bonds[1] != -1:
            #     # 说明是有两条边的情况 判断其应该用哪个进行扩增
            #     if (self.bond_idx == 0):
            #         data.aug_bond = 0
            #     else:
            #         data.aug_bond = 1
            # elif (data.bonds[0] != -1):  # 设计为随机选择一个
            #     data = randrop(data, self.aug_ratio)
            #     data.aug_bond = 0
            # else:  # data.bonds[1]!=-1
            #     data = randrop(data, self.aug_ratio)
            #     data.aug_bond = 1

            # 对数据进行处理 因为是两组数据 对应两个正例
            # 如果分子可能断开的键的数目>=2个，则随机选取两个 一个作为前面的dataset 一个作为后面的dataset
            # 如果只有一个键
        # num_bonds_changes = data.num_bonds.item()
        # if num_bonds_changes == 0:
        #     pass
        # elif num_bonds_changes == 1:
        #     # 对图进行增强 边仅使用这一个 无法对边进行增强
        #     data = randrop(data, self.aug_ratio)
        #     data.bond_aug = data.bonds[0]
        # else:  # !=1 为0排除
        #     # 随机对边和图进行增强——根据数据集的比例来看，为1的占很多，因此这里最好全部为对边进行增强
        #     id = np.random.choice([1])
        #     # id = np.random.choice([0, 1])
        #     if id == 0:
        #         # 对图进行增强
        #         # 但是输入的内容是有两个的，如果对应的bond==0如何处理？这样也不会保证两次都是使用同一种增强方式？
        #         data = randrop(data, self.aug_ratio)
        #         data.bond_aug = data.bonds[0]  # 应当是随机选择一个键，两种情况下都是选择相同的键，如果使用随机选择，则不能保证下一个也是选择这个
        #     else:
        #         # 对边进行增强 随机选择一个边 作为对应的边的信息 新增一个信息作为其对应的键？
        #         bonds_list_odd = list(i for i in range(len(data.bonds)) if i % 2 != 0)
        #         bonds_list_even = list(i for i in range(len(data.bonds)) if i % 2 == 0)
        #         # 如果为0，则在所有的偶数里进行选择 如果为1，则在所有的奇数里进行选择 这样能够保证其不会重复
        #         bond_idx = 0
        #         if self.bond_idx == 0:
        #             try:
        #                 bond_idx = np.random.choice(bonds_list_even)
        #             except:
        #                 print('empty!')
        #         else:
        #             bond_idx = np.random.choice(bonds_list_odd)
        #         data.bond_aug = data.bonds[bond_idx]
        # if self.aug_ratio == 0:
        #     pass
        # else:
        #     data = randrop(data, aug_ratio=self.aug_ratio)
        return data

    @property
    def raw_file_names(self):
        # file_name_list = os.listdir(self.raw)
        # assert len(file_name_list) == 1     # currently assume we have a
        # # single raw file
        return ['{}.pickle'.format(self.dataset)]  # raw_paths调用此

    # 调用父类初始化时会优先执行@property函数
    @property
    def processed_file_names(self):  # 返回一个处理后的文件列表，包含processed_dir中的文件目录，根据列表内容决定哪些数据已经处理过从而可以跳过处理；
        # return 'geometric_data_full_new_processed.pt'  # 预处理之后保存到此处 好像是先执行的此处，之后才执行process这个函数
        # 如果没找到这个文件则会执行process获得处理后的文件
        return ['{}.pt'.format(self.dataset)]

    def download(self):
        pass

    # 修改版本，改动使得其能够满足在self.collate()之后得到结果
    def process(self):
        # 修改模型为只要是改变的都作为reaction center
        # https://www.cxymm.net/article/qq_32583189/110196193
        data_smiles_list = []
        data_list = []
        # atom_mapping, reactants, products, bonds_new, bonds_changed, bonds_lost= _load_uspto_dataset(self.raw_paths[0])#注意这里如果文件名不进行更改的话是有问题的
        reaction = open(self.raw_paths[0], 'rb')
        # 这里不能只是这一个数据，不然不管是什么任务都是这个
        reaction_data = pickle.load(reaction)
        # num_bonds_changed = reaction_data['num_changed']
        # bonds_changed = reaction_data['bonds_changed']
        # reaction_data = reaction_data[reaction_data['num_changed'] >= 1]
        # 保存
        # new_reaction_data.to_csv('dataset/raw/USPTO/data/new_uspto_mit.csv')
        products = reaction_data['products']
        bonds_changed = reaction_data['bonds_changes']
        num_bonds_changed = reaction_data['num_changed']
        # 对数据进行处理
        num_bonds_changed.mask(num_bonds_changed == 9, 8, inplace=True)
        num_bonds_changed.mask(num_bonds_changed == 10, 9, inplace=True)
        num_bonds_changed.mask(num_bonds_changed == 11, 10, inplace=True)
        num_bonds_changed.mask(num_bonds_changed == 13, 11, inplace=True)
        num_bonds_changed.mask(num_bonds_changed == 15, 12, inplace=True)
        num_bonds_changed.mask(num_bonds_changed == 17, 13, inplace=True)
        for i in trange(len(products)):
            # idx=alter_idx[i]
            try:
                rdkit_mol = Chem.MolFromSmiles(products[i])
                # data = mol2graph(rdkit_mol)
                data = mol_to_graph_data(rdkit_mol)
                data.num_bonds = torch.tensor(num_bonds_changed[i])
                if num_bonds_changed[i] == 0:
                    data.bonds = [(-1, -1, -1)]
                else:
                    data.bonds = bonds_changed[i]
                data_list.append(data)
                data_smiles_list.append(products[i])
            except:
                print('exception')

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        data, slices = self.collate(data_list)  # 注意将这里的数据转换为一定的格式，不然无法用这个进行处理
        torch.save((data, slices), self.processed_paths[0])  # 保存geometric_data_processed.pt文件


class USPTOPairedDataset(InMemoryDataset):
    def __init__(self,
                 root,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 dataset=None,
                 aug='none', aug_ratio=None):
        self.dataset = dataset
        self.aug = aug
        self.aug_ratio = aug_ratio
        self.get_count = 0
        self.transform, self.pre_transform, self.pre_filter = transform, pre_transform, pre_filter
        super(USPTOPairedDataset, self).__init__(root, transform, pre_transform,
                                                 pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    def get(self, idx):
        data = Data()
        for key in self.data.keys:
            if key != 'y':
                item, slices = self.data[key], self.slices[key]
                s = list(repeat(slice(None), item.dim()))
                s[data.__cat_dim__(key, item)] = slice(slices[idx], slices[
                    idx + 1])
                data[key] = item[s]
            else:
                item, slices = self.data[key], self.slices[key]
                data[key] = item[idx]
        return data

    @property
    def raw_file_names(self):
        return ['{}.pickle'.format(self.dataset)]  # raw_paths调用此

    @property
    def processed_file_names(self):
        return ['{}_bond_features.pt'.format(self.dataset)]

    def download(self):
        pass

    def process(self):
        # 修改模型为只要是改变的都作为reaction center
        # https://www.cxymm.net/article/qq_32583189/110196193
        # data_smiles_list = []
        data_list = []
        # atom_mapping, reactants, products, bonds_new, bonds_changed, bonds_lost= _load_uspto_dataset(self.raw_paths[0])#注意这里如果文件名不进行更改的话是有问题的
        reaction = open(self.raw_paths[0], 'rb')
        # 这里不能只是这一个数据，不然不管是什么任务都是这个
        reaction_data = pickle.load(reaction)
        # num_bonds_changed = reaction_data['num_changed']
        # bonds_changed = reaction_data['bonds_changed']
        # reaction_data = reaction_data[reaction_data['num_changed'] >= 1]
        # 保存
        # new_reaction_data.to_csv('dataset/raw/USPTO/data/new_uspto_mit.csv')
        products = reaction_data['products']
        bonds_changed_mol = reaction_data['bonds_changes']
        # num_bonds_changed = reaction_data['num_changed']
        # 对数据进行处理
        # prod_mol = Chem.MolFromSmiles(products)
        for i, smi_products in tqdm(enumerate(list(products))):
            mol_products = Chem.MolFromSmiles(smi_products)
            # num_bonds = mol_products.GetNumBonds()
            bonds_list = mol_products.GetBonds()
            bonds_changed = bonds_changed_mol[i]
            data = mol_to_graph_data(mol_products)
            bond_idx = []
            data.label = 0
            for bond_changed in bonds_changed:
                bond_idx.append(bond_changed[2])
            # atom_idx_list = []
            # bonds_feat = []
            for bond in bonds_list:
                # data = Data()
                # 传入label，即其是否是断开的位点
                if bond.GetIdx() in bond_idx:
                    bond_feat = Local_FromSmiles(smi_products, bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), 2)
                    # bonds_feat.append(list(bond_feat.values()))
                    data.bond_feature = torch.tensor(list(bond_feat.values()), dtype=torch.float32)
                    data.y = 1
                    data_list.append(data.clone())
                else:
                    if data.label == 0:
                        bond_feat = Local_FromSmiles(smi_products, bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), 2)
                        # bonds_feat.append(list(bond_feat.values()))
                        data.bond_feature = torch.tensor(list(bond_feat.values()), dtype=torch.float32)
                        data.label = 1
                        data.y = 0
                        data_list.append(data.clone())
                    else:
                        continue
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        data, slices = self.collate(data_list)  # 注意将这里的数据转换为一定的格式，不然无法用这个进行处理
        torch.save((data, slices), self.processed_paths[0])  # 保存geometric_data_processed.pt文件

class SupervisedUSPTODataset(InMemoryDataset):
    def __init__(self,
                 root,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 dataset=None,
                 aug='none', aug_ratio=None):
        self.dataset = dataset
        self.aug = aug
        self.aug_ratio = aug_ratio
        self.get_count = 0
        self.transform, self.pre_transform, self.pre_filter = transform, pre_transform, pre_filter
        super(SupervisedUSPTODataset, self).__init__(root, transform, pre_transform,
                                                 pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    def get(self, idx):
        data = Data()
        for key in self.data.keys:
            if key != 'y':
                item, slices = self.data[key], self.slices[key]
                s = list(repeat(slice(None), item.dim()))
                s[data.__cat_dim__(key, item)] = slice(slices[idx], slices[
                    idx + 1])
                data[key] = item[s]
            else:
                item, slices = self.data[key], self.slices[key]
                data[key] = item[idx]
        return data

    @property
    def raw_file_names(self):
        return ['{}.pickle'.format(self.dataset)]  # raw_paths调用此

    @property
    def processed_file_names(self):
        return ['{}_supervised.pt'.format(self.dataset)]

    def download(self):
        pass

    def process(self):
        # 修改模型为只要是改变的都作为reaction center
        # https://www.cxymm.net/article/qq_32583189/110196193
        # data_smiles_list = []
        data_list = []
        # atom_mapping, reactants, products, bonds_new, bonds_changed, bonds_lost= _load_uspto_dataset(self.raw_paths[0])#注意这里如果文件名不进行更改的话是有问题的
        reaction = open(self.raw_paths[0], 'rb')
        # 这里不能只是这一个数据，不然不管是什么任务都是这个
        reaction_data = pickle.load(reaction)
        # num_bonds_changed = reaction_data['num_changed']
        # bonds_changed = reaction_data['bonds_changed']
        # reaction_data = reaction_data[reaction_data['num_changed'] >= 1]
        # 保存
        # new_reaction_data.to_csv('dataset/raw/USPTO/data/new_uspto_mit.csv')
        products = reaction_data['products']
        bonds_changed_mol = reaction_data['bonds_changes']
        # num_bonds_changed = reaction_data['num_changed']
        # 对数据进行处理
        # prod_mol = Chem.MolFromSmiles(products)
        for i, smi_products in tqdm(enumerate(list(products))):
            mol_products = Chem.MolFromSmiles(smi_products)
            # num_bonds = mol_products.GetNumBonds()
            bonds_list = mol_products.GetBonds()
            bonds_changed = bonds_changed_mol[i]
            data = mol_to_graph_data(mol_products)
            bond_idx = []
            data.label = 0
            for bond_changed in bonds_changed:
                bond_idx.append(bond_changed[2])
            # atom_idx_list = []
            # bonds_feat = []
            for bond in bonds_list:
                # data = Data()
                # 传入label，即其是否是断开的位点
                if bond.GetIdx() in bond_idx:
                    bond_feat = Local_FromSmiles(smi_products, bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), 2)
                    # bonds_feat.append(list(bond_feat.values()))
                    data.bond_feature = torch.tensor(list(bond_feat.values()), dtype=torch.float32)
                    data.y = 1
                    data_list.append(data.clone())
                else:
                    if data.label == 0:
                        bond_feat = Local_FromSmiles(smi_products, bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), 2)
                        # bonds_feat.append(list(bond_feat.values()))
                        data.bond_feature = torch.tensor(list(bond_feat.values()), dtype=torch.float32)
                        data.label = 1
                        data.y = 0
                        data_list.append(data.clone())
                    else:
                        continue
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        data, slices = self.collate(data_list)  # 注意将这里的数据转换为一定的格式，不然无法用这个进行处理
        torch.save((data, slices), self.processed_paths[0])  # 保存geometric_data_processed.pt文件

class USPTOFPDataset(InMemoryDataset):
    def __init__(self,
                 root,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 dataset=None,
                 aug='none', aug_ratio=None):
        self.dataset = dataset
        self.aug = aug
        self.aug_ratio = aug_ratio
        self.get_count = 0
        self.transform, self.pre_transform, self.pre_filter = transform, pre_transform, pre_filter
        super(USPTOFPDataset, self).__init__(root, transform, pre_transform,
                                             pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    def get(self, idx):
        data = Data()
        for key in self.data.keys:
            if key != 'prod':
                item, slices = self.data[key], self.slices[key]
                s = list(repeat(slice(None), item.dim()))
                s[data.__cat_dim__(key, item)] = slice(slices[idx], slices[
                    idx + 1])
                data[key] = item[s]
            else:
                item, slices = self.data[key], self.slices[key]
                data[key] = item[idx]
        return data

    @property
    def raw_file_names(self):
        return ['{}.pickle'.format(self.dataset)]  # raw_paths调用此

    @property
    def processed_file_names(self):
        return ['{}_FP_ml_bonds_full.pt'.format(self.dataset)]

    def download(self):
        pass

    def process(self):
        data_list = []
        # atom_mapping, reactants, products, bonds_new, bonds_changed, bonds_lost= _load_uspto_dataset(self.raw_paths[0])#注意这里如果文件名不进行更改的话是有问题的
        reaction = open(self.raw_paths[0], 'rb')
        # 这里不能只是这一个数据，不然不管是什么任务都是这个
        reaction_data = pickle.load(reaction)
        products = reaction_data['products']
        bonds_changed_mol = reaction_data['bonds_changes']
        for i, smi_products in tqdm(enumerate(list(products))):
            mol_products = Chem.MolFromSmiles(smi_products)
            bonds_list = mol_products.GetBonds()
            bonds_changed = bonds_changed_mol[i]
            data = mol_to_fp_data(mol_products)
            # data.prod = smi_products
            data.label = 0
            bond_idx = []
            data.num_bonds = len(bonds_changed)
            for bond_changed in bonds_changed:
                bond_idx.append(bond_changed[2])
            for bond in bonds_list:
                # data = Data()
                # 传入label，即其是否是断开的位点
                if bond.GetIdx() in bond_idx:
                    data.y = 1  # 会覆盖原来的值！！！
                    bond_feat =  Local_FromSmiles(smi_products, bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), 2)
                    data.bond_feature = torch.tensor(list(bond_feat.values()), dtype=torch.float32)
                    data_list.append(data.clone())
                else:
                    if data.label == 0:
                        data.y = 0
                        data.label = 1
                        bond_feat = Local_FromSmiles(smi_products, bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), 2)
                        data.bond_feature = torch.tensor(list(bond_feat.values()), dtype=torch.float32)
                        data_list.append(data.clone())
                    else:
                        continue
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        data, slices = self.collate(data_list)  # 注意将这里的数据转换为一定的格式，不然无法用这个进行处理
        torch.save((data, slices), self.processed_paths[0])  # 保存geometric_data_processed.pt文件


class ComplexityDataset(InMemoryDataset):
    def __init__(self,
                 root,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 dataset=None,
                 aug='none', aug_ratio=None):
        self.dataset = dataset
        self.aug = aug
        self.aug_ratio = aug_ratio
        self.get_count = 0
        self.transform, self.pre_transform, self.pre_filter = transform, pre_transform, pre_filter
        super(ComplexityDataset, self).__init__(root, transform, pre_transform,
                                                pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    def get(self, idx):
        data = Data()
        for key in self.data.keys:
            if key != 'product' and key != 'bonds' and key != 'num_bonds' and key != 'y':
                item, slices = self.data[key], self.slices[key]
                s = list(repeat(slice(None), item.dim()))
                s[data.__cat_dim__(key, item)] = slice(slices[idx], slices[
                    idx + 1])
                data[key] = item[s]
            else:
                item, slices = self.data[key], self.slices[key]
                data[key] = item[idx]
        return data

    @property
    def raw_file_names(self):
        return ['{}.csv'.format(self.dataset)]

    @property
    def processed_file_names(self):
        return ['{}.pt'.format(self.dataset)]

    def download(self):
        pass

    def process(self):
        data_list = []
        folder = 'complexity/raw'
        file_path = os.path.join(folder, self.raw_file_names[0])
        complexity_data = pd.read_csv(file_path, usecols=[2, 3])
        data_len = len(complexity_data)
        smiles_data = complexity_data['SMILES']
        label_data = complexity_data['meanComplexity']
        for i in trange(data_len):
            try:
                mol_products = Chem.MolFromSmiles(smiles_data[i])
                data = mol_to_graph_data(mol_products)
                data.y = label_data[i]
                data_list.append(data)
            except:
                print(smiles_data[i])

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])