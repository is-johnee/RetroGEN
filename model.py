import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, softmax
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
import torch.nn.functional as F
import torch.nn as nn
from torch_scatter import scatter_add
from torch_geometric.nn.inits import glorot, zeros
from reactive_site_features import Local_FromSmiles
from mol_encoder import AtomEncoder, BondEncoder, BondLocalEncoder
from rdkit import Chem
from torch_geometric.nn.inits import glorot, reset
# from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader

num_atom_type = 120  # including the extra mask tokens=119
num_chirality_tag = 4  # original =3. including the extra mask tokens=3 !!!
num_possible_degree = 12
num_formal_charge = 12
num_numH = 10
num_radical_e = 6
num_possible_hybridization = 6
num_is_aromatic = 2
num_is_in_ring = 2

num_bond_type = 6  # including aromatic and self-loop edge, and extra masked tokens
num_bond_stereo = 6  # original =3, inlcuding the extra mask tokens=3
num_is_conjugated = 2
num_bond_direction = 3


class GINConv(MessagePassing):
    """
    Extension of GIN aggregation to incorporate edge information by concatenation.

    Args:
        emb_dim (int): dimensionality of embeddings for nodes and edges.
        embed_input (bool): whether to embed input or not.


    See https://arxiv.org/abs/1810.00826
    """

    def __init__(self, emb_dim, aggr="add"):
        super(GINConv, self).__init__()
        # multi-layer perceptron
        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2 * emb_dim), torch.nn.ReLU(),
                                       torch.nn.Linear(2 * emb_dim, emb_dim))
        self.bond_encoder = BondEncoder(emb_dim)
        self.aggr = aggr

    # propgate
    def forward(self, x, edge_index, edge_attr):
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:, 0] = 6
        self_loop_attr[:, 1] = 3
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)
        edge_embeddings = self.bond_encoder(edge_attr)
        return self.propagate(edge_index[0], x=x, edge_attr=edge_embeddings)
    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return self.mlp(aggr_out)


class GCNConv(MessagePassing):

    def __init__(self, emb_dim, aggr="add"):
        super(GCNConv, self).__init__()

        self.emb_dim = emb_dim
        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.bond_encoder = BondEncoder(emb_dim)
        self.aggr = aggr

    def norm(self, edge_index, num_nodes, dtype):
        ### assuming that self-loops have been already added in edge_index
        edge_weight = torch.ones((edge_index[0].size(1),), dtype=dtype,
                                 device=edge_index[0].device)
        row, col = edge_index[0]
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, edge_attr):
        # add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:, 0] = 6
        self_loop_attr[:, 1] = 3
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)
        edge_embeddings = self.bond_encoder(edge_attr)
        norm = self.norm(edge_index, x.size(0), x.dtype)

        x = self.linear(x)

        return self.propagate(edge_index[0], x=x, edge_attr=edge_embeddings, norm=norm)

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * (x_j + edge_attr)


class GATConv(MessagePassing):
    def __init__(self, emb_dim, heads=2, negative_slope=0.2, aggr="add", concat=False, bias=True):
        super(GATConv, self).__init__()

        self.aggr = aggr
        self.concat = concat
        self.emb_dim = emb_dim
        self.heads = heads
        self.negative_slope = negative_slope

        self.weight_linear = torch.nn.Linear(emb_dim, heads * emb_dim)
        self.att = torch.nn.Parameter(torch.Tensor(1, heads, 2 * emb_dim))

        # self.bias = torch.nn.Parameter(torch.Tensor(emb_dim))

        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(self.heads * self.emb_dim))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(self.emb_dim))
        else:
            self.register_parameter('bias', None)

        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, heads * emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, heads * emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.att)
        zeros(self.bias)

    def forward(self, x, edge_index, edge_attr):
        # add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))

        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:, 0] = 4  # bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(edge_attr[:, 1])
        # edge_embeddings = edge_embeddings.view(-1, self.heads, self.emb_dim)
        # x = self.weight_linear(x).view(-1, self.heads, self.emb_dim)
        x = self.weight_linear(x)
        return self.propagate(edge_index[0], x=x, edge_attr=edge_embeddings)

    def message(self, edge_index, x_i, x_j, edge_attr):
        x_i = x_i.view(-1, self.heads, self.emb_dim)
        x_j = x_j.view(-1, self.heads, self.emb_dim)
        edge_attr = edge_attr.view(-1, self.heads, self.emb_dim)
        x_j += edge_attr

        alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index[0])
        alpha = alpha.view(-1, self.heads, 1)
        # return (x_j * alpha).view(-1, self.heads, 1)
        # return (x_j * alpha).view(x_j.size(0), -1)
        return (x_j * alpha.view(-1, self.heads, 1)).view(-1, self.heads * self.emb_dim)

    def update(self, aggr_out):
        aggr_out = aggr_out.view(-1, self.heads, self.emb_dim).mean(dim=1)
        aggr_out = aggr_out + self.bias
        return aggr_out


class GraphSAGEConv(MessagePassing):
    def __init__(self, emb_dim, aggr="mean"):
        super(GraphSAGEConv, self).__init__()

        self.emb_dim = emb_dim
        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

        self.aggr = aggr

    def forward(self, x, edge_index, edge_attr):
        # add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))

        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:, 0] = 4  # bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(edge_attr[:, 1])

        x = self.linear(x)

        return self.propagate(edge_index[0], x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return F.normalize(aggr_out, p=2, dim=-1)


class GNN(torch.nn.Module):
    """


    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        JK (str): last, concat, max or sum.
        max_pool_layer (int): the layer from which we use max pool rather than add pool for neighbor aggregation
        drop_ratio (float): dropout rate
        gnn_type: gin, gcn, graphsage, gat

    Output:
        node representations

    """

    def __init__(self, num_layer, emb_dim, JK="last", drop_ratio=0, gnn_type="gin"):
        super(GNN, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.atom_encoder = AtomEncoder(emb_dim)
        self.bond_encoder = BondEncoder(emb_dim)
        # self.feature_attention = FeatureAttention(emb_dim, 2)
        # self.feature_attention.reset_parameters()
        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")
        ###List of MLPs
        self.gnns = torch.nn.ModuleList()
        for layer in range(num_layer):
            if gnn_type == "gin":
                self.gnns.append(GINConv(emb_dim, aggr="add"))
            elif gnn_type == "gcn":
                self.gnns.append(GCNConv(emb_dim))
            elif gnn_type == "gat":
                self.gnns.append(GATConv(emb_dim))
            elif gnn_type == "graphsage":
                self.gnns.append(GraphSAGEConv(emb_dim))

        ##List of batchnorms
        self.batch_norms = torch.nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

    # def forward(self, x, edge_index, edge_attr):
    def forward(self, *argv):
        if len(argv) == 4:
            x, edge_index, edge_attr, batch = argv[0], argv[1], argv[2], argv[3]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        else:
            raise ValueError("unmatched number of arguments.")
        x = self.atom_encoder(x)
        h_list = [x]
        for layer in range(self.num_layer):
            h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
            # h = self.batch_norms[layer](h)
            if layer == self.num_layer - 1:
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)
            h_list.append(h)
        ### Different implementations of Jk-concat
        if self.JK == "concat":
            node_representation = torch.cat(h_list, dim=1)
        elif self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "max":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.max(torch.cat(h_list, dim=0), dim=0)[0]
        elif self.JK == "sum":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list, dim=0), dim=0)[0]

        return node_representation

    def reset_parameters(self):
        pass


class GNN_graphCL(torch.nn.Module):
    """
    Extension of GIN to incorporate edge information by concatenation.

    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        num_tasks (int): number of tasks in multi-task learning scenario
        drop_ratio (float): dropout rate
        JK (str): last, concat, max or sum.
        graph_pooling (str): sum, mean, max, attention, set2set
        gnn_type: gin, gcn, graphsage, gat

    See https://arxiv.org/abs/1810.00826
    JK-net: https://arxiv.org/abs/1806.03536
    """

    def __init__(self, num_layer, model_device, emb_dim, bond_dim, JK="last", drop_ratio=0, graph_pooling="mean",
                 gnn_type="gin"):
        super(GNN_graphCL, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.device = model_device
        self.emb_dim = emb_dim
        self.bond_local_emb = BondLocalEncoder(bond_dim, self.device)
        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.gnn = GNN(num_layer, emb_dim, JK, drop_ratio, gnn_type=gnn_type)
        self.proj_head = nn.Sequential(nn.Linear(emb_dim, emb_dim * 2),
                                       nn.ReLU(inplace=True),
                                       nn.Linear(emb_dim * 2, emb_dim))
        self.bond_proj = nn.Sequential(nn.Linear(bond_dim, bond_dim * 2),
                                       nn.ReLU(inplace=True),
                                       nn.Linear(bond_dim * 2, bond_dim))
        self.concat_feature = nn.Sequential(nn.Linear(emb_dim + bond_dim, (emb_dim + bond_dim) * 2),
                                            nn.ReLU(inplace=True),
                                            nn.Linear((emb_dim + bond_dim) * 2, emb_dim + bond_dim))
        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        elif graph_pooling == "attention":
            if self.JK == "concat":
                self.pool = GlobalAttention(gate_nn=torch.nn.Linear((self.num_layer + 1) * emb_dim, 1))
            else:
                self.pool = GlobalAttention(gate_nn=torch.nn.Linear(emb_dim, 1))
        elif graph_pooling[:-1] == "set2set":
            set2set_iter = int(graph_pooling[-1])
            if self.JK == "concat":
                self.pool = Set2Set((self.num_layer + 1) * emb_dim, set2set_iter)
            else:
                self.pool = Set2Set(emb_dim, set2set_iter)
        else:
            raise ValueError("Invalid graph pooling type.")

        # For graph-level binary classification
        if graph_pooling[:-1] == "set2set":
            self.mult = 2
        else:
            self.mult = 1

        if self.JK == "concat":
            self.graph_pred_linear = torch.nn.Linear(self.mult * self.emb_dim,
                                                     2)
        else:
            self.graph_pred_linear = torch.nn.Linear(self.mult * self.emb_dim, 2)

    def from_pretrained(self, model_file):
        self.load_state_dict(torch.load(model_file))

    def forward(self, *argv):
        if len(argv) == 4:
            x, edge_index, edge_attr, batch = argv[0], argv[1], argv[2], argv[3]
        elif len(argv) == 1:
            data = argv[0]  # DataBatch
            data_batch = DataLoader(data, batch_size=len(data))
            batched_motif = next(iter(data_batch))
            # batched_motif = Batch.from_data_list(data)
            batched_motif.to(self.device)
            x, edge_index, edge_attr, batch = batched_motif.x, batched_motif.edge_index, batched_motif.edge_attr, batched_motif.batch
        else:
            raise ValueError("unmatched number of arguments.")
        node_representation = self.gnn(x, edge_index, edge_attr, batch)
        graph_representation = self.pool(node_representation, batch)
        return batched_motif, node_representation, self.proj_head(graph_representation)

    def bond_info(self, smiles_list, bond_list):
        begin_atom_idx_list = []
        end_atom_idx_list = []
        for i, smi in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(smi)
            begin_atom_idx_list.append(mol.GetBondWithIdx(bond_list[i]).GetBeginAtomIdx())
            end_atom_idx_list.append(mol.GetBondWithIdx(bond_list[i]).GetEndAtomIdx())
        bonds_features = []
        for i in range(len(smiles_list)):
            feat = Local_FromSmiles(smiles_list[i], begin_atom_idx_list[i], end_atom_idx_list[i], 2)
            bonds_features.append(list(feat.values()))
        bonds_features_tensor = torch.tensor(bonds_features, dtype=torch.float32)
        features_bonds = self.bond_local_emb(bonds_features_tensor.to(self.device))
        return self.bond_proj(features_bonds)

    def loss_cl(self, x1, x2):
        T = 0.5
        batch_size, _ = x1.size()  # torch.Size([32, 128])
        # similarity matrix
        x1_abs = x1.norm(dim=1)
        x2_abs = x2.norm(dim=1)
        sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
        sim_matrix = torch.exp(sim_matrix / T)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]  # Nx1对角线元素
        # left view
        loss_left = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)  # Nx1/Nx1
        loss_left = - torch.log(loss_left).mean()

        # right view
        loss_right = pos_sim / (sim_matrix.sum(dim=0) - pos_sim)
        loss_right = - torch.log(loss_right).mean()

        # loss
        loss = (loss_left + loss_right) / 2.0
        return loss

    def reset_parameters(self):
        pass


class GNN_graphpred(torch.nn.Module):
    """
    Extension of GIN to incorporate edge information by concatenation.

    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        num_tasks (int): number of tasks in multi-task learning scenario
        drop_ratio (float): dropout rate
        JK (str): last, concat, max or sum.
        graph_pooling (str): sum, mean, max, attention, set2set
        gnn_type: gin, gcn, graphsage, gat

    See https://arxiv.org/abs/1810.00826
    JK-net: https://arxiv.org/abs/1806.03536
    """

    def __init__(self, num_layer, emb_dim, JK="last", drop_ratio=0, graph_pooling="mean", gnn_type="gin"):
        super(GNN_graphpred, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.gnn = GNN(num_layer, emb_dim, JK, drop_ratio, gnn_type=gnn_type)
        # self.gnn = GNN(num_layer, emb_dim, JK, drop_ratio, gnn_type=gnn_type)

        # Different kind of graph pooling
        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        elif graph_pooling == "attention":
            if self.JK == "concat":
                self.pool = GlobalAttention(gate_nn=torch.nn.Linear((self.num_layer + 1) * emb_dim, 1))
            else:
                self.pool = GlobalAttention(gate_nn=torch.nn.Linear(emb_dim, 1))
        elif graph_pooling[:-1] == "set2set":
            set2set_iter = int(graph_pooling[-1])
            if self.JK == "concat":
                self.pool = Set2Set((self.num_layer + 1) * emb_dim, set2set_iter)
            else:
                self.pool = Set2Set(emb_dim, set2set_iter)
        else:
            raise ValueError("Invalid graph pooling type.")

        # For graph-level binary classification
        if graph_pooling[:-1] == "set2set":
            self.mult = 2
        else:
            self.mult = 1

        if self.JK == "concat":
            self.graph_pred_linear = torch.nn.Linear(self.mult * (self.num_layer + 1) * self.emb_dim, 1)
        else:
            self.graph_pred_linear = torch.nn.Linear(self.mult * self.emb_dim, 1)

    def from_pretrained(self, model_file):
        # self.gnn = GNN(self.num_layer, self.emb_dim, JK = self.JK, drop_ratio = self.drop_ratio)
        self.load_state_dict(torch.load(model_file))

    def forward(self, *argv):
        if len(argv) == 4:
            x, edge_index, edge_attr, batch = argv[0], argv[1], argv[2], argv[3]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        else:
            raise ValueError("unmatched number of arguments.")

        node_representation = self.gnn(x, edge_index, edge_attr)

        return self.graph_pred_linear(self.pool(node_representation, batch))


if __name__ == "__main__":
    # pass
    def loss_cl(x1, x2):
        T = 1.0
        batch_size, _ = x1.size()

        # similarity matrix
        x1_abs = x1.norm(dim=1)
        print('x1_abs: ', x1_abs)
        x2_abs = x2.norm(dim=1)
        print('x2_abs: ', x2_abs)
        sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
        sim_matrix = torch.exp(sim_matrix / T)
        print('sim_matrix: ', sim_matrix)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]  # Nx1
        print('pos_sim: ', pos_sim)

        print('row sum: ', sim_matrix.sum(dim=1))
        print('col sum: ', sim_matrix.sum(dim=0))

        # left view
        loss_left = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)  # Nx1/Nx1
        print('loss_left: ', loss_left)
        loss_left = - torch.log(loss_left).mean()

        # right view
        loss_right = pos_sim / (sim_matrix.sum(dim=0) - pos_sim)
        print('loss_right: ', loss_right)
        loss_right = - torch.log(loss_right).mean()

        # loss
        loss = (loss_left + loss_right) / 2.0
        print(loss)
        return loss


    x1 = torch.tensor([[1, 0, 1], [1, 0, 2], [1, 1, 1]], dtype=torch.float32)
    x2 = torch.tensor([[1, 2, 3], [1, 1, 2], [1, 3, 1]], dtype=torch.float32)
    loss_cl(x1, x2)
    loss_cl(x2, x1)
