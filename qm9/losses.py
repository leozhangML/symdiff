import torch


def sum_except_batch(x):
    return x.view(x.size(0), -1).sum(dim=-1)


def assert_correctly_masked(variable, node_mask):
    assert (variable * (1 - node_mask)).abs().sum().item() < 1e-8


def compute_loss_and_nll(args, generative_model, nodes_dist, x, h, node_mask, edge_mask, context):
    bs, n_nodes, n_dims = x.size()

    if args.probabilistic_model == 'diffusion':
        edge_mask = edge_mask.view(bs, n_nodes * n_nodes)  # [bs, n_nodes^2] - different to vdm?

        assert_correctly_masked(x, node_mask)

        # Here x is a position tensor, and h is a dictionary with keys
        # 'categorical' and 'integer'.
        # [bs, n_nodes, 3]; dict of cat, int: [bs, n_nodes, num_classes] [bs, n_nodes, 1]; [bs, n_nodes, 1]; [bs, n_nodes^2]; [bs, n_nodes, context_nf]
        nll = generative_model(x, h, node_mask, edge_mask, context)

        N = node_mask.squeeze(2).sum(1).long()  # [bs]

        log_pN = nodes_dist.log_prob(N) if args.molecule else torch.zeros_like(nll)

        assert nll.size() == log_pN.size()
        nll = nll - log_pN  # include likelihood of the node number

        # Average over batch.
        nll = nll.mean(0)

        reg_term = torch.tensor([0.]).to(nll.device)
        mean_abs_z = 0.
    else:
        raise ValueError(args.probabilistic_model)

    return nll, reg_term, mean_abs_z
