from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from DGCNConv import DGCNConv


def decoder(Z):
    A_pred = torch.sigmoid((torch.matmul(Z,Z.t())))
    return A_pred


class DGCN_link_prediction(torch.nn.Module):

    def __init__(self, num_features: int, hidden: int, label_dim: int, dropout: Optional[float] = None,
                 improved: bool = False, cached: bool = False):
        super(DGCN_link_prediction, self).__init__()
        self.dropout = dropout
        self.dgconv = DGCNConv(improved=improved, cached=cached)
        self.linear = nn.Linear(hidden*6, label_dim)

        self.lin1 = nn.Linear(num_features,hidden,bias=False)
        self.lin2 = nn.Linear(hidden*3, hidden, bias=False)
        self.lin3 = nn.Linear(hidden*3,hidden,bias=False)

        self.bias1 = nn.Parameter(torch.Tensor(1, hidden))
        self.bias2 = nn.Parameter(torch.Tensor(1, hidden))
        self.bias3 = nn.Parameter(torch.Tensor(1, hidden))

        nn.init.zeros_(self.bias1)
        nn.init.zeros_(self.bias2)
        nn.init.zeros_(self.bias3)

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.lin3.reset_parameters()
        nn.init.zeros_(self.bias1)
        nn.init.zeros_(self.bias2)
        nn.init.zeros_(self.bias3)
        self.linear.reset_parameters()

    def forward(self, x: torch.FloatTensor, edge_index: torch.LongTensor,
                edge_in: torch.LongTensor, edge_out: torch.LongTensor,
                query_edges: torch.LongTensor,
                in_w: Optional[torch.FloatTensor] = None, out_w: Optional[torch.FloatTensor] = None) -> torch.FloatTensor:

        x = self.lin1(x)
        x1 = self.dgconv(x,edge_index)
        x2 = self.dgconv(x,edge_in,in_w)
        x3 = self.dgconv(x,edge_out,out_w)

        x1 += self.bias1
        x2 += self.bias1
        x3 += self.bias1

        x = torch.cat((x1, x2, x3),axis=-1)
        x = F.relu(x)

        x = self.lin2(x)
        x1 = self.dgconv(x, edge_index)
        x2 = self.dgconv(x, edge_in, in_w)
        x3 = self.dgconv(x, edge_out, out_w)

        x1 += self.bias2
        x2 += self.bias2
        x3 += self.bias2

        x = torch.cat((x1, x2, x3), axis=-1)
        x = F.relu(x)

        x = self.lin3(x)
        x1 = self.dgconv(x, edge_index)
        x2 = self.dgconv(x, edge_in, in_w)
        x3 = self.dgconv(x, edge_out, out_w)

        x1 += self.bias3
        x2 += self.bias3
        x3 += self.bias3

        x = torch.cat((x1, x2, x3), axis=-1)
        x = F.relu(x)
        x = torch.cat((x[query_edges[:, 0]], x[query_edges[:, 1]]), dim=-1)

        if self.dropout > 0:
            x = F.dropout(x, self.dropout, training=self.training)
        x = self.linear(x)
        x = F.softmax(x,dim=1)
        return x