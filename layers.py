import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter

import numpy as np


class GraphConvolution(Module):

    def __init__(self, in_features, out_features, act=F.relu):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.act = act
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        self.weight = self.get_init_weight([self.in_features, self.out_features])


    def get_init_weight(self, shape):
        bound = np.sqrt(6.0 / (np.sum(shape)))
        ini = torch.rand(shape) * 2 * bound - bound
        return torch.nn.Parameter(ini, requires_grad=True)

    def forward(self, input, adj):
        tmp = torch.spmm(adj, input)
        output = torch.mm(tmp, self.weight)
        output = self.act(output)
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'



class ClusterLayer(nn.Module):

    def __init__(self, num_clusters, hidden_dim, alpha=1):
        super(ClusterLayer, self).__init__()
        self.alpha = alpha
        self.network = Parameter(torch.Tensor(num_clusters, hidden_dim)).float()
        torch.nn.init.xavier_normal_(self.network.data)

    def forward(self, z) -> torch.Tensor:
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.network, 2), 2) / self.alpha)
        q = q.pow((self.alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        return q

class Discriminator(nn.Module):

    def __init__(self, hidden1, hidden2):
        super(Discriminator, self).__init__()
        self.bil = nn.Bilinear(hidden1, hidden2, 1)
        self.act = nn.Sigmoid()
        self.get_init_weight()

    def get_init_weight(self):
        torch.nn.init.xavier_uniform_(self.bil.weight.data)
        if self.bil.bias is not None:
            self.bil.bias.data.fill_(0.0)

    def forward(self, h_1, h_2, sample_idx):
        emb_1 = torch.squeeze(self.bil(h_2, h_1),1)
        emb_1 = self.act(emb_1)
        emb_2_list = []
        for i in range(len(sample_idx)):
            h_sam = h_2[sample_idx[i]]
            emb_2_item = self.bil(h_sam, h_1)
            emb_2_list.append(emb_2_item)
        emb_2 = torch.squeeze(torch.stack(emb_2_list,1),0)
        emb_2 = self.act(emb_2)

        return emb_1, emb_2
