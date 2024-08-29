import torch
import torch.nn as nn
from egnn.egnn_new import EGNN, GNN
from equivariant_diffusion.utils import remove_mean, remove_mean_with_mask
import numpy as np


class GNNEnc(nn.Module):
    def __init__(self, in_node_nf, context_node_nf,
                 out_node_nf,
                 n_dims, hidden_nf=64, device='cpu',
                 act_fn=torch.nn.SiLU(), n_layers=4, attention=False,
                 normalization_factor=100, aggregation_method='sum'):
        
        """
        normalization_factor: for aggregation_method
        in_node_nf is with time
        """

        super().__init__()
        self.gnn = GNN(
            in_node_nf=in_node_nf + context_node_nf + 3, in_edge_nf=0,
            hidden_nf=hidden_nf, out_node_nf=out_node_nf, device=device,
            act_fn=act_fn, n_layers=n_layers, attention=attention,
            normalization_factor=normalization_factor, aggregation_method=aggregation_method)

        self.context_node_nf = context_node_nf  # 0 for no conditioning
        self.device = device
        self.n_dims = n_dims
        self._edges_dict = {}

    def forward(self, t, xh, node_mask, edge_mask, context=None):
        raise NotImplementedError

    def _forward(self, t, xh, node_mask, edge_mask, context):  # used in phi for training
        """
        t: [bs] or [1]
        xh: [bs, n_nodes, dims] - contains pos, cat h, int h
        node_mask: [bs, n_nodes]
        edge_masks [bs*n_nodes^2, 1]
        context: ?

        Just need to accept [bs, n_nodes, dim] for xh (add conditioning info) and accept node_mask for
        correct masking - need some projection matrix for correct dims?

        """
        bs, n_nodes, dims = xh.shape
        h_dims = dims - self.n_dims
        edges = self.get_adj_matrix(n_nodes, bs, self.device)  # needed for GNN
        edges = [x.to(self.device) for x in edges]
        node_mask = node_mask.view(bs*n_nodes, 1)
        edge_mask = edge_mask.view(bs*n_nodes*n_nodes, 1) 
        xh = xh.view(bs*n_nodes, -1).clone() * node_mask  # [bs*n_nodes, dims]  -  why use .clone() here?
        x = xh[:, 0:self.n_dims].clone()  # [bs*n_nodes, n_dim] for pos
        h = xh[:, self.n_dims:].clone()  # [bs*n_nodes, h_dim]

        # t is different over the batch dimension.
        h_time = t.view(bs, 1).repeat(1, n_nodes)
        h_time = h_time.view(bs * n_nodes, 1)
        h = torch.cat([h, h_time], dim=1)  # add time to h

        if context is not None:
            # We're conditioning, awesome!
            context = context.view(bs*n_nodes, self.context_node_nf)  # [bs*n_nodes, context_node_nf]
            h = torch.cat([h, context], dim=1)  # add context to h

        xh = torch.cat([x, h], dim=1)
        # linear embedding, then message passing layers then linear projection
        output = self.gnn(xh, edges, node_mask=node_mask)

        return output.view(bs, n_nodes, -1)

    def get_adj_matrix(self, n_nodes, batch_size, device):
        if n_nodes in self._edges_dict:
            edges_dic_b = self._edges_dict[n_nodes]
            if batch_size in edges_dic_b:
                return edges_dic_b[batch_size]
            else:
                # get edges for a single sample
                rows, cols = [], []
                for batch_idx in range(batch_size):
                    for i in range(n_nodes):
                        for j in range(n_nodes):
                            rows.append(i + batch_idx * n_nodes)
                            cols.append(j + batch_idx * n_nodes)
                edges = [torch.LongTensor(rows).to(device),
                         torch.LongTensor(cols).to(device)]
                edges_dic_b[batch_size] = edges
                return edges
        else:
            self._edges_dict[n_nodes] = {}
            return self.get_adj_matrix(n_nodes, batch_size, device)
