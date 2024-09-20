import numpy as np

import torch
import torch.nn as nn

from equivariant_diffusion.utils import assert_correctly_masked, assert_mean_zero_with_mask


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

"""Implementation of equivariance metrics"""

def convert_x_to_xh(args, x, h, eval_model):
    if args.molecule:
        xh = torch.cat([x, h['categorical'], h['integer']], dim=-1)  # [bs, n_nodes, dims]
    else:
        bs, n_nodes, _ = x.shape
        xh = torch.cat([x, torch.zeros(bs, n_nodes, eval_model.in_node_nf, device=x.device)], dim=-1)
    return xh

def add_noise(args, x, h, node_mask, eval_model, sample_t0=False):
    # x: [bs, n_nodes, n_dims]
    # h: [bs, n_nodes, in_node_nf]
    # node_mask: [bs, n_nodes, 1]

    xh = convert_x_to_xh(args, x, h, eval_model)

    if sample_t0:
        t = torch.zeros(len(x), 1, device=x.device)
        gamma_0 = eval_model.inflate_batch_array(eval_model.gamma(t), x)
        alpha_0 = eval_model.alpha(gamma_0, x)
        sigma_0 = eval_model.sigma(gamma_0, x)

        # Sample z_0 given x, h for timestep t, from q(z_t | x, h)
        eps_0 = eval_model.sample_combined_position_feature_noise(
            n_samples=x.size(0), n_nodes=x.size(1), node_mask=node_mask, com_free=args.com_free)
        z_t = alpha_0 * xh + sigma_0 * eps_0

        return z_t, eps_0

    else:
        t_int = torch.randint(
            1, eval_model.T + 1, size=(x.size(0), 1), device=x.device).float()  # [bs, 1] from [0, T]
        s_int = t_int - 1

        # Normalize t to [0, 1]. Note that the negative
        # step of s will never be used, since then p(x | z0) is computed.
        s = s_int / eval_model.T
        t = t_int / eval_model.T

        # Compute gamma_t via the network.
        gamma_t = eval_model.inflate_batch_array(eval_model.gamma(t), x)

        # Compute alpha_t and sigma_t from gamma.
        alpha_t = eval_model.alpha(gamma_t, x)  # [bs, 1, 1]
        sigma_t = eval_model.sigma(gamma_t, x)

        # Sample zt ~ Normal(alpha_t x, sigma_t)
        eps = eval_model.sample_combined_position_feature_noise(
            n_samples=x.size(0), n_nodes=x.size(1), node_mask=node_mask, com_free=args.com_free)  # [bs, n_nodes, dims] - masks out non-atom indexes

        # Sample z_t given x, h for timestep t, from q(z_t | x, h)
        z_t = alpha_t * xh + sigma_t * eps

    return z_t, s, t

def batch_inputs(t, xh, node_mask, n):
    # t: [bs, 1]
    # xh_t: [bs, n_nodes, dims]

    _, n_nodes, dims = xh.shape

    repeated_t = t.unsqueeze(1).repeat_interleave(n, dim=1).reshape(-1, 1)  # [bs*n, 1]
    repeated_xh = xh.unsqueeze(1).repeat_interleave(n, dim=1).reshape(-1, n_nodes, dims)  # [bs*n, n_nodes, dims]
    repeated_node_mask = node_mask.unsqueeze(1).repeat_interleave(n, dim=1).reshape(-1, n_nodes, 1)  # [bs*n, n_nodes, 1]

    return repeated_t, repeated_xh, repeated_node_mask

def compute_normal_log_probs(sample, mu, sigma, node_mask, eval_model, use_xh=False):
    # return log_prob of Gaussian
    # sample: [bs, n_nodes, dims]
    assert_correctly_masked(mu, node_mask)
    assert_mean_zero_with_mask(mu, node_mask)
    assert_correctly_masked(sample, node_mask)
    assert_mean_zero_with_mask(sample, node_mask)   
    number_of_nodes = torch.sum(node_mask.squeeze(2), dim=1)
    eff_n_nodes = number_of_nodes - 1 if eval_model.com_free else number_of_nodes
    if use_xh and eval_model.molecule:
        d = eff_n_nodes * (eval_model.n_dims + eval_model.in_node_nf)
    else:
        d = eff_n_nodes * eval_model.n_dims
    square_norm = torch.sum((sample - mu).pow(2), dim=(1, 2))  # [bs]
    sigma = sigma.reshape(-1)
    return -0.5 * (square_norm / (sigma**2) + d * (np.log(2*np.pi) + torch.log(sigma**2)))

def iwae_nll_0(args, xh, xh_0, eps, node_mask, eval_model, n_importance_samples):

    t_zeros = torch.zeros(len(xh_0), 1, device=xh_0.device)

    repeated_t_zeros, repeated_xh, repeated_node_mask = batch_inputs(
        t_zeros, xh, node_mask, n_importance_samples)

    _, repeated_eps, _ = batch_inputs(t_zeros, eps, node_mask, n_importance_samples)
    _, repeated_xh_0, _ = batch_inputs(t_zeros, xh_0, node_mask, n_importance_samples)
    
    repeated_gamma_0 = eval_model.gamma(repeated_t_zeros).unsqueeze(-1)  # [bs*n_importance_samples, 1, 1]
    repeated_h = repeated_xh[:, :, args.n_dims:]
    repeated_net_out = eval_model.phi(repeated_xh_0, repeated_t_zeros, 
                                      repeated_node_mask, None, None)

    log_probs = eval_model.log_pxh_given_z0_without_constants(
        None, repeated_h, repeated_xh_0, repeated_gamma_0, repeated_eps, 
        repeated_net_out, repeated_node_mask
    ).reshape(-1, n_importance_samples)

    log_probs_constants = eval_model.log_constants_p_x_given_z0(
        repeated_xh_0[:, :, :args.n_dims], repeated_node_mask
    ).reshape(-1, n_importance_samples)

    iwae = torch.logsumexp(log_probs+log_probs_constants, dim=-1) - np.log(n_importance_samples)  # [bs]

    return -iwae

def iwae_equivariance(args, s, xh_s, t, xh_t, node_mask, eval_model, n_importance_samples, 
                      sample_t0=False, use_xh=False):
    # compute iwae estimate of the full model equivariance
    # only needs x part of xh_s
    # s: [bs, 1]
    # xh_s: [bs, n_nodes, dims]

    repeated_s, repeated_xh_s, repeated_node_mask = batch_inputs(s, xh_s, node_mask, n_importance_samples)
    repeated_t, repeated_xh_t, _ = batch_inputs(t, xh_t, node_mask, n_importance_samples)

    # [bs*n_importance_samples, n_nodes, dims], [bs*n_importance_samples, 1, 1]
    if sample_t0:
        mu, sigma = eval_model.sample_p_xh_given_z0(
            repeated_xh_t, repeated_node_mask, None, None,  
            remove_noise=True)
    else:
        mu, sigma = eval_model.sample_p_zs_given_zt(
            repeated_s, repeated_t, repeated_xh_t, 
            repeated_node_mask, None, None, remove_noise=True)

    if use_xh:
        # [bs, n_importance_samples]
        log_probs = compute_normal_log_probs(
            repeated_xh_s, mu, sigma, repeated_node_mask, 
            eval_model, use_xh=True
        ).reshape(-1, n_importance_samples)
    else:
        # [bs*n_importance_samples, n_nodes, 3]
        repeated_x_s = repeated_xh_s[:, :, :args.n_dims]
        mu_x = mu[:, :, :args.n_dims]

        # [bs, n_importance_samples]
        log_probs = compute_normal_log_probs(
            repeated_x_s, mu_x, sigma, repeated_node_mask, eval_model
        ).reshape(-1, n_importance_samples)

    return torch.logsumexp(log_probs, dim=-1) - np.log(n_importance_samples)  # [bs]

def apply_g(args, xh, g):
    # xh: [bs, n_nodes, dims]
    return torch.cat(
        [torch.bmm(xh[:, :, :args.n_dims], g.transpose(1, 2)),
         xh[:, :, args.n_dims:]], dim=-1
    )

def compute_equivariance_metrics_model(args, x, h, node_mask, eval_model, n_importance_samples):
    # h is {'categorical': one_hot, 'integer': charges} for args.molecule as we use this within test

    bs, n_nodes, _ = x.shape
    x, h, _ = eval_model.normalize(x, h, node_mask)

    z_t, s, t = add_noise(args, x, h, node_mask, eval_model, sample_t0=False) # [bs, n_nodes, dims], [bs, 1, 1], [bs, 1]
    z_0, _ = add_noise(args, x, h, node_mask, eval_model, sample_t0=True)

    z_s = eval_model.sample_p_zs_given_zt(s, t, z_t, node_mask, None, None)  # [bs, n_nodes, dims]
    x, _ = eval_model.sample_p_xh_given_z0(z_0, node_mask, None, None)  # new x

    g = orthogonal_haar(dim=args.n_dims, target_tensor=torch.empty(bs, device=x.device))  # [bs, n_dims, n_dims]
    g_z_t = apply_g(args, z_t, g)
    g_z_s = apply_g(args, z_s, g)

    g_z_0 = apply_g(args, z_0, g)
    g_x = apply_g(args, torch.cat([x, torch.zeros(bs, n_nodes, eval_model.in_node_nf, device=x.device)], dim=-1), g)

    iwae_z = iwae_equivariance(args, s, z_s, t, z_t, node_mask, eval_model, n_importance_samples)  # [bs]
    iwae_g_z = iwae_equivariance(args, s, g_z_s, t, g_z_t, node_mask, eval_model, n_importance_samples)

    empty_t = torch.empty_like(s, device=s.device)
    iwae_z_0 = iwae_equivariance(args, empty_t, x, empty_t, z_0, node_mask, eval_model, 
                                 n_importance_samples, sample_t0=True)
    iwae_g_z_0 = iwae_equivariance(args, empty_t, g_x, empty_t, g_z_0, node_mask, eval_model, 
                                   n_importance_samples, sample_t0=True)

    metric = eval_model.T * (iwae_z - iwae_g_z) + (iwae_z_0 - iwae_g_z_0)  # [bs]

    return metric.mean().item()  # average over batch

def compute_equivariance_metrics_backbone(args, x, h, node_mask, eval_model):

    bs, n_nodes, _ = x.shape
    x, h, _ = eval_model.normalize(x, h, node_mask)

    z_t, s, t = add_noise(args, x, h, node_mask, eval_model, sample_t0=False) # [bs, n_nodes, dims], [bs, 1, 1], [bs, 1]
    z_0, _ = add_noise(args, x, h, node_mask, eval_model, sample_t0=True)

    g = orthogonal_haar(dim=args.n_dims, target_tensor=torch.empty(bs, device=x.device))  # [bs, n_dims, n_dims]
    g_z_t = apply_g(args, z_t, g)
    g_z_0 = apply_g(args, z_0, g)

    # [bs, n_nodes, dims], [bs, 1, 1]
    mu_z_t, sigma_t = eval_model.sample_p_zs_given_zt(s, t, z_t, node_mask, None, None, 
                                                      remove_noise=True, model="backbone")
    mu_g_z_t, _ = eval_model.sample_p_zs_given_zt(s, t, g_z_t, node_mask, None, None, 
                                                  remove_noise=True, model="backbone")

    mu_z_0, sigma_0 = eval_model.sample_p_xh_given_z0(z_0, node_mask, None, None, 
                                                      remove_noise=True, model="backbone")
    mu_g_z_0, _ = eval_model.sample_p_xh_given_z0(g_z_0, node_mask, None, None, 
                                                  remove_noise=True, model="backbone")

    # [bs, n_nodes, n_dims]
    mu_x_t = mu_z_t[:, :, :args.n_dims]
    g_inv_mu_g_x_t = torch.bmm(mu_g_z_t[:, :, :args.n_dims], g)

    mu_x_0 = mu_z_0[:, :, :args.n_dims]
    g_inv_mu_g_x_0 = torch.bmm(mu_g_z_0[:, :, :args.n_dims], g)

    # [bs]
    kl_t = torch.sum((mu_x_t - g_inv_mu_g_x_t).pow(2), dim=(1, 2)) / (2 * sigma_t.reshape(-1)**2)
    kl_0 = torch.sum((mu_x_0 - g_inv_mu_g_x_0).pow(2), dim=(1, 2)) / (2 * sigma_0.reshape(-1)**2)

    metric = eval_model.T * kl_t + kl_0

    return metric.mean().item()

def compute_normal_entropy(args, sigma, node_mask, eval_model):
    # sigma: [bs, 1, 1]
    num_of_nodes = torch.sum(node_mask.squeeze(2), dim=1)  # [bs]
    eff_n_nodes = num_of_nodes - 1 if eval_model.com_free else num_of_nodes
    if args.molecule:
        d = eff_n_nodes * (args.n_dims + eval_model.in_node_nf)  # doesn't work for conditioning
    else:
        d = eff_n_nodes * args.n_dims
    return d * (np.log(2*np.pi*np.e) + torch.log(sigma.reshape(-1)**2)) / 2  # [bs]

def iwae_nll(args, x, h, node_mask, eval_model, n_importance_samples, nodes_dist=None):

    x, h, delta_log_px = eval_model.normalize(x, h, node_mask)

    if not args.molecule:
        delta_log_px = torch.zeros_like(delta_log_px)

    z_t, s, t = add_noise(args, x, h, node_mask, eval_model, sample_t0=False) # [bs, n_nodes, dims], [bs, 1, 1], [bs, 1]
    z_0, eps = add_noise(args, x, h, node_mask, eval_model, sample_t0=True)

    # compute prior term
    xh = convert_x_to_xh(args, x, h, eval_model)
    kl_prior = eval_model.kl_prior(xh, node_mask)  # [bs]

    # compute terms for t \ne 0, x0 option is for q(x_{t-1}|x_t, x_0)
    z_s = eval_model.sample_p_zs_given_zt(s, t, z_t, node_mask, 
                                          None, None, x0=xh)
    _, sigma_t = eval_model.sample_p_zs_given_zt(s, t, z_t, node_mask, None, 
                                                 None, remove_noise=True, x0=xh)

    entropy_s_given_t = compute_normal_entropy(args, sigma_t, node_mask, eval_model)
    iwae_z = iwae_equivariance(args, s, z_s, t, z_t, node_mask, eval_model, 
                               n_importance_samples, sample_t0=False, use_xh=True)

    nll_t = entropy_s_given_t + iwae_z  # [bs]

    # compute terms for t=0
    nll_0 = iwae_nll_0(args, xh, z_0, eps, node_mask, eval_model, n_importance_samples)

    nll = kl_prior + eval_model.T * nll_t + nll_0 - delta_log_px  # [bs]

    if nodes_dist is not None:
        N = node_mask.squeeze(2).sum(1).long()  # [bs]
        nll -= nodes_dist.log_prob(N) if args.molecule else torch.zeros_like(nll)

    return nll.mean().item()
