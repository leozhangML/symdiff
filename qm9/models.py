import torch
from torch.distributions.categorical import Categorical

import numpy as np
from egnn.models import EGNN_dynamics_QM9

from equivariant_diffusion.en_diffusion import EnVariationalDiffusion
from sym_nn.sym_nn import SymDiffPerceiver_dynamics, SymDiffTransformer_dynamics


def get_model(args, device, dataset_info, dataloader_train):
    histogram = dataset_info['n_nodes']  # e.g. qm9, 9: 83366 etc.
    in_node_nf = len(dataset_info['atom_decoder']) + int(args.include_charges)  # 'atom_decoder': ['H', 'C', 'N', 'O', 'F']; \pm 1
    nodes_dist = DistributionNodes(histogram)  # will sample over all nodes

    prop_dist = None
    if len(args.conditioning) > 0:
        prop_dist = DistributionProperty(dataloader_train, args.conditioning)

    if args.condition_time:
        dynamics_in_node_nf = in_node_nf + 1  # add time conditioning, use one dim for modelling charges
    else:
        print('Warning: dynamics model is _not_ conditioned on time.')
        dynamics_in_node_nf = in_node_nf

    if args.model == "egnn_dynamics" or args.model == "gnn_dynamics":

        net_dynamics = EGNN_dynamics_QM9(
            in_node_nf=dynamics_in_node_nf, context_node_nf=args.context_node_nf,
            n_dims=3, device=device, hidden_nf=args.nf,
            act_fn=torch.nn.SiLU(), n_layers=args.n_layers,
            attention=args.attention, tanh=args.tanh, mode=args.model, norm_constant=args.norm_constant,
            inv_sublayers=args.inv_sublayers, sin_embedding=args.sin_embedding,
            normalization_factor=args.normalization_factor, aggregation_method=args.aggregation_method)
    
    elif args.model == "symdiff_perceiver_dynamics":

        net_dynamics = SymDiffPerceiver_dynamics(
            args,
            in_node_nf=in_node_nf,
            context_node_nf=args.context_node_nf,

            gamma_num_latents=args.gamma_num_latents, 
            gamma_d_latents=args.gamma_d_latents,
            gamma_n_pad=args.gamma_n_pad,
            gamma_num_blocks=args.gamma_num_blocks, 
            gamma_num_self_attends_per_block=args.gamma_num_self_attends_per_block, 
            gamma_num_self_attention_heads=args.gamma_num_self_attention_heads, 
            gamma_num_cross_attention_heads=args.gamma_num_cross_attention_heads,
            gamma_attention_probs_dropout_prob=args.gamma_attention_probs_dropout_prob,
            gamma_pos_num_channels=args.gamma_pos_num_channels,
            gamma_num_heads=args.gamma_num_heads,

            k_num_latents=args.k_num_latents,
            k_d_latents=args.k_d_latents,
            k_n_pad=args.k_n_pad,
            k_num_blocks=args.k_num_blocks,
            k_num_self_attends_per_block=args.k_num_self_attends_per_block,
            k_num_self_attention_heads=args.k_num_self_attention_heads,
            k_num_cross_attention_heads=args.k_num_cross_attention_heads,
            k_attention_probs_dropout_prob=args.k_attention_probs_dropout_prob,
            k_pos_num_channels=args.k_pos_num_channels,
            k_num_heads=args.k_num_heads,
            k_decoder_self_attention=args.k_decoder_self_attention,
            k_num_self_heads=args.k_num_self_heads,
            num_bands=args.num_bands,
            max_resolution=args.max_resolution,
            concat_t=args.concat_t,
            device=device)

    elif args.model == "symdiff_transformer_dynamics":

        net_dynamics = SymDiffTransformer_dynamics(
            args,
            in_node_nf=in_node_nf,
            context_node_nf=args.context_node_nf, 

            gamma_num_enc_layers=args.gamma_num_enc_layers,
            gamma_num_dec_layers=args.gamma_num_dec_layers,
            gamma_d_model=args.gamma_d_model, 
            gamma_nhead=args.gamma_nhead,
            gamma_dim_feedforward=args.gamma_dim_feedforward, 
            gamma_dropout=args.gamma_dropout,

            k_num_layers=args.k_num_layers,
            k_d_model=args.k_d_model, 
            k_nhead=args.k_nhead,
            k_dim_feedforward=args.k_dim_feedforward, 
            k_dropout=args.k_dropout,

            num_bands=args.num_bands,
            max_resolution=args.max_resolution,
            t_fourier=args.t_fourier,
            concat_t=args.concat_t,
        )

    if args.probabilistic_model == 'diffusion':
        vdm = EnVariationalDiffusion(
            dynamics=net_dynamics,
            in_node_nf=in_node_nf,
            n_dims=3,
            timesteps=args.diffusion_steps,
            noise_schedule=args.diffusion_noise_schedule,
            noise_precision=args.diffusion_noise_precision,
            loss_type=args.diffusion_loss_type,
            norm_values=args.normalize_factors,
            include_charges=args.include_charges
            )

        return vdm, nodes_dist, prop_dist

    else:
        raise ValueError(args.probabilistic_model)


def get_optim(args, generative_model):
    optim = torch.optim.AdamW(
        generative_model.parameters(),
        lr=args.lr, amsgrad=True,
        weight_decay=1e-12)

    return optim


class DistributionNodes:
    def __init__(self, histogram):

        self.n_nodes = []
        prob = []
        self.keys = {}
        for i, nodes in enumerate(histogram):  # nodes goes through keys  - i.e. 1 - 9
            self.n_nodes.append(nodes)
            self.keys[nodes] = i  # stores pos index of nodes
            prob.append(histogram[nodes])  # stores freq of nodes
        self.n_nodes = torch.tensor(self.n_nodes)
        prob = np.array(prob)
        prob = prob/np.sum(prob)

        self.prob = torch.from_numpy(prob).float()

        entropy = torch.sum(self.prob * torch.log(self.prob + 1e-30))
        print("Entropy of n_nodes: H[N]", entropy.item())  # to python scalar

        self.m = Categorical(torch.tensor(prob))  # distribution over n_nodes

    def sample(self, n_samples=1):
        idx = self.m.sample((n_samples,))  # [n_samples]
        return self.n_nodes[idx]

    def log_prob(self, batch_n_nodes):
        assert len(batch_n_nodes.size()) == 1

        idcs = [self.keys[i.item()] for i in batch_n_nodes]
        idcs = torch.tensor(idcs).to(batch_n_nodes.device)

        log_p = torch.log(self.prob + 1e-30)

        log_p = log_p.to(batch_n_nodes.device)

        log_probs = log_p[idcs]  # multiple for each instance?

        return log_probs


class DistributionProperty:
    def __init__(self, dataloader, properties, num_bins=1000, normalizer=None):  # uses defaults for kwargs
        self.num_bins = num_bins
        self.distributions = {}  # over conditioning args
        self.properties = properties  # from args - e.g.  homo | lumo | alpha | gap | mu | Cv
        for prop in properties:
            self.distributions[prop] = {}
            self._create_prob_dist(dataloader.dataset.data['num_atoms'],
                                   dataloader.dataset.data[prop],
                                   self.distributions[prop])

        self.normalizer = normalizer

    def set_normalizer(self, normalizer):
        self.normalizer = normalizer

    def _create_prob_dist(self, nodes_arr, values, distribution):
        min_nodes, max_nodes = torch.min(nodes_arr), torch.max(nodes_arr)
        for n_nodes in range(int(min_nodes), int(max_nodes) + 1):
            idxs = nodes_arr == n_nodes
            values_filtered = values[idxs]
            if len(values_filtered) > 0:
                probs, params = self._create_prob_given_nodes(values_filtered)  # for each prop and then n_nodes, set prob
                distribution[n_nodes] = {'probs': probs, 'params': params}

    def _create_prob_given_nodes(self, values):
        n_bins = self.num_bins #min(self.num_bins, len(values))
        prop_min, prop_max = torch.min(values), torch.max(values)
        prop_range = prop_max - prop_min + 1e-12
        histogram = torch.zeros(n_bins)
        for val in values:
            i = int((val - prop_min)/prop_range * n_bins)  # in [0, 1]
            # Because of numerical precision, one sample can fall in bin int(n_bins) instead of int(n_bins-1)
            # We move it to bin int(n_bind-1 if tat happens)
            if i == n_bins:
                i = n_bins - 1
            histogram[i] += 1
        probs = histogram / torch.sum(histogram)
        probs = Categorical(torch.tensor(probs))  # over binned values that prop can take (given n_nodes)
        params = [prop_min, prop_max]  # range of prob values (given_n_nodes)
        return probs, params

    def normalize_tensor(self, tensor, prop):
        assert self.normalizer is not None
        mean = self.normalizer[prop]['mean']
        mad = self.normalizer[prop]['mad']
        return (tensor - mean) / mad

    def sample(self, n_nodes=19):
        vals = []
        for prop in self.properties:
            dist = self.distributions[prop][n_nodes]
            idx = dist['probs'].sample((1,))  # in [0, 1]
            val = self._idx2value(idx, dist['params'], len(dist['probs'].probs))  # returns back out the corresponding val
            val = self.normalize_tensor(val, prop)  # ensures the normalised range for the prop
            vals.append(val)
        vals = torch.cat(vals)
        return vals

    def sample_batch(self, nodesxsample):
        vals = []
        for n_nodes in nodesxsample:
            vals.append(self.sample(int(n_nodes)).unsqueeze(0))  # kwarg??
        vals = torch.cat(vals, dim=0)
        return vals

    def _idx2value(self, idx, params, n_bins):
        prop_range = params[1] - params[0]
        left = float(idx) / n_bins * prop_range + params[0]
        right = float(idx + 1) / n_bins * prop_range + params[0]
        val = torch.rand(1) * (right - left) + left  # sampling uniform in bin range
        return val


if __name__ == '__main__':
    dist_nodes = DistributionNodes()
    print(dist_nodes.n_nodes)
    print(dist_nodes.prob)
    for i in range(10):
        print(dist_nodes.sample())
