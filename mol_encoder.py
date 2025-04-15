import torch

full_atom_feature_dims = [120, 3]
# full_atom_feature_dims = [120, 5, 12, 12, 10, 6, 8, 3, 3]
# [119, 4, 11, 11, 9, 5, 7, 2, 2]
full_bond_feature_dims = [7, 4]
# full_bond_feature_dims = [5, 4, 7, 3]
full_bond_local_feature_dims = [21, 21, 17, 20, 14, 21, 8, 6, 23, 22, 5, 3, 21, 20, 22, 20, 7, 4, 4, 4, 8, 9, 6, 6]


# full_bond_local_feature_dims = [21, 21, 17, 20, 11, 21, 8, 6, 23, 22, 600, 5, 3, 21, 20, 22, 20, 7, 4, 4, 4, 8, 8, 4, 4]

class AtomEncoder(torch.nn.Module):

    def __init__(self, emb_dim):
        super(AtomEncoder, self).__init__()

        self.atom_embedding_list = torch.nn.ModuleList()

        for i, dim in enumerate(full_atom_feature_dims):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)

    def forward(self, x):
        x_embedding = 0
        for i in range(x.shape[1]):
            x_embedding += self.atom_embedding_list[i](x[:, i])

        return x_embedding


class BondEncoder(torch.nn.Module):

    def __init__(self, emb_dim):
        super(BondEncoder, self).__init__()

        self.bond_embedding_list = torch.nn.ModuleList()

        for i, dim in enumerate(full_bond_feature_dims):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.bond_embedding_list.append(emb)

    def forward(self, edge_attr):
        bond_embedding = 0
        for i in range(edge_attr.shape[1]):
            bond_embedding += self.bond_embedding_list[i](edge_attr[:, i])

        return bond_embedding


class BondLocalEncoder(torch.nn.Module):
    def __init__(self, emb_dim, device):
        super(BondLocalEncoder, self).__init__()
        self.bond_local_list = torch.nn.ModuleList()
        self.device = device
        for i, dim in enumerate(full_bond_local_feature_dims):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.bond_local_list.append(emb)

    def forward(self, local_attr):
        local_emb = 0
        local_attr = local_attr.long().to(self.device)
        try:
            for i in range(local_attr.shape[1]):
                local_emb += self.bond_local_list[i](local_attr[:, i])
        except:
            print(local_attr[:, i])
            print('index_error')
        return local_emb