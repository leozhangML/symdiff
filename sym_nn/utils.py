import numpy as np

import torch
import torch.nn as nn


def qr(inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    assert inputs.ndim == 3 and inputs.shape[1] == inputs.shape[2]

    # NOTE: the current implementation of torch.linalg.qr can be numerically
    # unstable during backwards pass when output has (close to) linearly
    # dependent columns, at least until pivoting is implemented upstream
    # (see comment torch.linalg.qr docs, as well as
    # https://github.com/pytorch/pytorch/issues/42792). Hence we convert to
    # double before applying the QR (and then convert back)
    #
    # NOTE: In addition, for some reason, QR decomposition on GPU is very
    # slow in PyTorch. This is a known issue: see
    # https://github.com/pytorch/pytorch/issues/22573). We work around this
    # as follows, although this is *still* much slower than it could be if
    # it were properly taking advantage of the GPU...
    #
    Q, R = torch.linalg.qr(inputs.cpu().double())
    Q = Q.to(torch.get_default_dtype()).to(inputs.device)
    R = R.to(torch.get_default_dtype()).to(inputs.device)

    # This makes sure the diagonal is positive, so that the Q matrix is
    # unique (and coresponds to the output produced by Gram-Schmidt, which
    # is equivariant)
    diag_sgns = torch.diag_embed(torch.diagonal(R, dim1=-2, dim2=-1).sign())

    # *Shouldn't* do anything but just to be safe:
    diag_sgns = diag_sgns.detach()

    return Q @ diag_sgns, diag_sgns @ R


def gram(x: torch.Tensor) -> torch.Tensor:
    x = x.to(torch.float64)  # TODO: Test without
    return x @ x.transpose(1, 2)


def cholesky(x: torch.Tensor) -> torch.Tensor:
    assert len(x.shape) == 3
    assert x.shape[1] == x.shape[2]
    result = torch.linalg.cholesky(x)
    return result.to(torch.get_default_dtype())  # TODO: Test without


# TODO: Just make this take 1 argument
def flatten(*args: torch.Tensor) -> torch.Tensor:
    return torch.cat([x.flatten(start_dim=1) for x in args], dim=1)


def make_square(x: torch.Tensor) -> torch.Tensor:
    assert len(x.shape) == 2

    d = np.sqrt(x.shape[1])

    assert d == int(d), "Input must be squareable"

    return x.reshape(x.shape[0], int(d), int(d))


def append(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    assert len(x.shape) == 3
    assert len(y.shape) == 2
    return torch.cat((x, y.unsqueeze(2)), dim=2)


def matmul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    assert len(x.shape) == 3
    assert len(y.shape) == 3
    assert x.shape[0] == y.shape[0]
    assert x.shape[2] == y.shape[1]
    return torch.bmm(x, y)


def transpose(x: torch.Tensor) -> torch.Tensor:
    assert len(x.shape) == 3
    return x.transpose(1, 2)


def orthogonal_haar(dim: int, target_tensor: torch.Tensor) -> torch.Tensor:
    """
    Implements the method of https://arxiv.org/pdf/math-ph/0609050v2.pdf
    (see (5.12) of that paper in particular)
    """

    noise = torch.randn(target_tensor.shape[0], dim, dim, 
                        device=target_tensor.device)
    return qr(noise)[0]

def compute_gradient_norm(model):
    grads = [
        param.grad.detach().flatten()
        for param in model.parameters()
        if param.grad is not None
    ]
    norm = torch.cat(grads).norm(p=2.0)
    return norm


"""NOTE: from https://github.com/lsj2408/Transformer-M/tree/main"""

@torch.jit.script
def gaussian(x, mean, std):
    pi = 3.14159
    a = (2*pi) ** 0.5
    return torch.exp(-0.5 * (((x - mean) / std) ** 2)) / (a * std)


class GaussianLayer(nn.Module):
    def __init__(self, K=128):
        super().__init__()
        self.K = K
        self.means = nn.Embedding(1, K)
        self.stds = nn.Embedding(1, K)
        nn.init.uniform_(self.means.weight, 0, 3)
        nn.init.uniform_(self.stds.weight, 0, 3)

    def forward(self, x, node_mask):
        # x: [bs, n_nodes, 3]
        # node_mask: [bs, n_nodes, 1]
        N = torch.sum(node_mask, dim=1, keepdim=True)  # [bs, 1, 1]
        dist_mat = torch.cdist(x, x).unsqueeze(-1)  # [bs, n_nodes, n_nodes]
        pos_emb = dist_mat.expand(-1, -1, -1, self.K)
        mean = self.means.weight.float().view(-1)
        std = self.stds.weight.float().view(-1).abs() + 1e-2
        pos_emb = gaussian(pos_emb.float(), mean, std).type_as(self.means.weight)  # [bs, n_nodes, n_nodes, K]
        pos_emb = pos_emb * node_mask[:, :, :, None] * node_mask[:, None, :, :]
        return pos_emb

def add_noise(args, x, h, node_mask, eval_model, sample_t0=False):
    # x: [bs, n_nodes, n_dims]
    # h: [bs, n_nodes, in_node_nf]
    # node_mask: [bs, n_nodes, 1]

    if args.molecule:
        xh = torch.cat([x, h['categorical'], h['integer']], dim=-1)  # [bs, n_nodes, dims]
    else:
        xh = torch.cat([x, torch.zeros_like(h, device=eval_model.device)], dim=-1)

    if sample_t0:
        t = torch.zeros(len(x), 1, device=eval_model.device)
        gamma_0 = eval_model.inflate_batch_array(eval_model.gamma(t), x)
        alpha_0 = eval_model.alpha(gamma_0, x)
        sigma_0 = eval_model.sigma(gamma_0, x)

        # Sample z_0 given x, h for timestep t, from q(z_t | x, h)
        eps_0 = eval_model.sample_combined_position_feature_noise(
            n_samples=x.size(0), n_nodes=x.size(1), node_mask=node_mask, com_free=args.com_free)
        z_t = alpha_0 * xh + sigma_0 * eps_0

        return z_t

    else:
        t_int = torch.randint(
            1, eval_model.T + 1, size=(x.size(0), 1), device=x.device).float()  # [bs, 1] from [0, T]
        s_int = t_int - 1

        # Normalize t to [0, 1]. Note that the negative
        # step of s will never be used, since then p(x | z0) is computed.
        s = s_int / eval_model.T
        t = t_int / eval_model.T

        # Compute gamma_s and gamma_t via the network.
        gamma_s = eval_model.inflate_batch_array(eval_model.gamma(s), x)
        gamma_t = eval_model.inflate_batch_array(eval_model.gamma(t), x)

        # Compute alpha_t and sigma_t from gamma.
        alpha_t = eval_model.alpha(gamma_t, x)  # [bs, 1, 1]
        sigma_t = eval_model.sigma(gamma_t, x)

        # Sample zt ~ Normal(alpha_t x, sigma_t)
        eps = eval_model.sample_combined_position_feature_noise(
            n_samples=x.size(0), n_nodes=x.size(1), node_mask=node_mask, com_free=args.com_free)  # [bs, n_nodes, dims] - masks out non-atom indexes

        # Sample z_t given x, h for timestep t, from q(z_t | x, h)
        z_t = alpha_t * xh + sigma_t * eps

    return z_t, gamma_s, gamma_t, s, t


def batched_inputs(t, xh, node_mask, n, eval_model, model="model"):
    # t: [bs, 1]
    # xh_t: [bs, n_nodes, dims]

    _, n_nodes, dims = xh.shape

    repeated_t = t.unsqueeze(1).repeat_interleave(n, dim=1).reshape(-1, 1)  # [bs*n, 1]
    repeated_xh = xh.unsqueeze(1).repeat_interleave(n, dim=1).reshape(-1, n_nodes, dims)  # [bs*n, n_nodes, dims]
    repeated_node_mask = node_mask.unsqueeze(1).repeat_interleave(n, dim=1).reshape(-1, n_nodes, 1)  # [bs*n, n_nodes, 1]

    """
    if model == "model":
        f_xh = eval_model.phi(
        repeated_xh, repeated_t,
        repeated_node_mask, None, None
    )

    elif model == "backbone":
        f_xh = eval_model.dynamics.k(
            repeated_t, repeated_xh[:, :, :args.n_dims], 
            repeated_xh[:, :, args.n_dims:], repeated_node_mask
        )
    return f_xh.reshape(bs, n, n_nodes, dims)
    """

    return repeated_t, repeated_xh, repeated_node_mask

def iwae_equivariance(args, s, xh_s, t, xh_t, node_mask, eval_model, n_importance_samples):
    # s: [bs, 1]
    # xh_s: [bs, n_nodes, dims]

    bs, n_nodes, dims = xh_s.shape

    repeated_s, repeated_xh_s, repeated_node_mask = batched_inputs(args, s, xh_s, node_mask, n_importance_samples, eval_model)
    repeated_t, repeated_xh_t, _ = batched_inputs(args, t, xh_t, node_mask, n_importance_samples, eval_model)

    # [bs*n_importance_samples, n_nodes, dims], [bs*n_importance_samples, 1, 1]
    mu, sigma = eval_model.sample_p_zs_given_zt(
        repeated_s, repeated_t, repeated_xh_t, 
        repeated_node_mask, None, None, remove_noise=True
    )

    sigma = sigma.reshape(bs, n_importance_samples, 1, 1)
    mu_x = mu[:, :, :args.n_dims].reshape(bs, n_importance_samples, n_nodes, args.n_dims)

    mean_diff = torch.sum((repeated_xh_s - mu)[:, :, :args.n_dims].pow(2), dim=(1, 2))












    f_xh_s = batched_inputs(s, xh_s, node_mask, n_importance_samples, eval_model)  # [bs, n_importance_samples, n_nodes, dims]
    f_xh_t = batched_inputs(t, xh_t, node_mask, n_importance_samples, eval_model)





    bs, n_nodes, dims = xh_s

    repeated_s = s.unsqueeze(1).repeat_interleave(n_importance_samples, dim=1)  # [bs, n_importance_samples, 1]
    repeated_xh_s = xh_s.unsqueeze(1).repeat_interleave(n_importance_samples, dim=1)  # [bs, n_importance_samples, n_nodes, dims]

    repeated_t = t.unsqueeze(1).repeat_interleave(n_importance_samples, dim=1)
    repeated_xh_t = xh_t.unsqueeze(1).repeat_interleave(n_importance_samples, dim=1)

    repeated_node_mask = node_mask.unsqueeze(1).repeat_interleave(n_importance_samples, dim=1)  # [bs, n_importance_samples, n_nodes, 1]

    f_xh_s = eval_model.phi(
        repeated_xh_s.reshape(-1, n_nodes, dims), repeated_s.reshape(-1, 1),
        repeated_node_mask.reshape(-1, n_nodes, 1), None, None
    )

    f_xh_s = eval_model.phi(
        repeated_xh_s.reshape(-1, n_nodes, dims), repeated_s.reshape(-1, 1),
        repeated_node_mask.reshape(-1, n_nodes, 1), None, None
    )


def compute_equivariance_metrics_model(args, x, h, node_mask, eval_model):
    # h is {'categorical': one_hot, 'integer': charges} for args.molecule as we use this within test

    z_t, gamma_s, gamma_t, s, t = add_noise(args, x, h, node_mask, eval_model, sample_t0=False) # [bs, n_nodes, dims], [bs, 1, 1], [bs, 1]
    z_s = eval_model.sample_p_zs_given_zt(s, t, z_t, node_mask, None, None)

    g = orthogonal_haar(dim=args.n_dims, target_tensor=torch.empty(len(x), device=eval_model.device))  # [bs, n_dims, n_dims]

    g_z_t = torch.cat(
        [torch.bmm(z_t[:, :, :args.n_dims], g.transpose(1, 2)),
         z_t[:, :, args.n_dims:]], dim=-1
    )

    g_s_t = torch.cat(
        [torch.bmm(z_s[:, :, :args.n_dims], g.transpose(1, 2)),
         z_s[:, :, args.n_dims:]], dim=-1
    )




    sigma2_t_given_s, sigma_t_given_s, alpha_t_given_s = \
            eval_model.sigma_and_alpha_t_given_s(gamma_t, gamma_s, z_t)
    sigma_s = eval_model.sigma(gamma_s, target_tensor=z_t)
    sigma_t = eval_model.sigma(gamma_t, target_tensor=z_t)




def compute_equivariance_metrics_backbone():
    pass