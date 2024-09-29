from equivariant_diffusion import utils
import numpy as np
import math
import torch
from egnn import models
from torch.nn import functional as F
from equivariant_diffusion import utils as diffusion_utils
import utils as data_aug_utils
from copy import copy, deepcopy

import sym_nn.utils as sym_nn_utils


# Defining some useful util functions.
def expm1(x: torch.Tensor) -> torch.Tensor:
    return torch.expm1(x)


def softplus(x: torch.Tensor) -> torch.Tensor:
    return F.softplus(x)


def sum_except_batch(x):
    return x.view(x.size(0), -1).sum(-1)  # from [bs, n_codes, dims] to [bs]


def clip_noise_schedule(alphas2, clip_value=0.001):
    """
    For a noise schedule given by alpha^2, this clips alpha_t / alpha_t-1. This may help improve stability during
    sampling.
    """
    alphas2 = np.concatenate([np.ones(1), alphas2], axis=0)

    alphas_step = (alphas2[1:] / alphas2[:-1])

    alphas_step = np.clip(alphas_step, a_min=clip_value, a_max=1.)
    alphas2 = np.cumprod(alphas_step, axis=0)

    return alphas2


def polynomial_schedule(timesteps: int, s=1e-4, power=3.):
    """
    A noise schedule based on a simple polynomial equation: 1 - x^power.
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas2 = (1 - np.power(x / steps, power))**2

    alphas2 = clip_noise_schedule(alphas2, clip_value=0.001)

    precision = 1 - 2 * s

    alphas2 = precision * alphas2 + s

    return alphas2


def cosine_beta_schedule(timesteps, s=0.008, raise_to_power: float = 1):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 2
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = np.clip(betas, a_min=0, a_max=0.999)
    alphas = 1. - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)

    if raise_to_power != 1:
        alphas_cumprod = np.power(alphas_cumprod, raise_to_power)

    return alphas_cumprod


def ve_schedule(timesteps, rho=3, sigma_min=1e-5, sigma_max=1):
    i = torch.linspace(timesteps-1, 0, timesteps)  # timesteps-1 to 0 in timesteps steps
    sigma_min_rho = sigma_min**(1/rho)
    sigma_max_rho = sigma_max**(1/rho)
    t_i = (sigma_max_rho + i/(timesteps-1) * (sigma_min_rho - sigma_max_rho))**(rho)
    return t_i  # 1D


def gaussian_entropy(mu, sigma):
    # In case sigma needed to be broadcast (which is very likely in this code).
    zeros = torch.zeros_like(mu)
    return sum_except_batch(
        zeros + 0.5 * torch.log(2 * np.pi * sigma**2) + 0.5
    )


def gaussian_KL(q_mu, q_sigma, p_mu, p_sigma, node_mask):
    """Computes the KL distance between two normal distributions.

        Args:
            q_mu: Mean of distribution q.
            q_sigma: Standard deviation of distribution q.
            p_mu: Mean of distribution p.
            p_sigma: Standard deviation of distribution p.
        Returns:
            The KL distance, summed over all dimensions except the batch dim.
        """
    return sum_except_batch(
            (
                torch.log(p_sigma / q_sigma)
                + 0.5 * (q_sigma**2 + (q_mu - p_mu)**2) / (p_sigma**2)
                - 0.5
            ) * node_mask
        )


def gaussian_KL_for_dimension(q_mu, q_sigma, p_mu, p_sigma, d):
    """Computes the KL distance between two normal distributions.

        Args:
            q_mu: Mean of distribution q.
            q_sigma: Standard deviation of distribution q.
            p_mu: Mean of distribution p.
            p_sigma: Standard deviation of distribution p.
        Returns:
            The KL distance, summed over all dimensions except the batch dim.
        """
    mu_norm2 = sum_except_batch((q_mu - p_mu)**2)
    assert len(q_sigma.size()) == 1
    assert len(p_sigma.size()) == 1
    return d * torch.log(p_sigma / q_sigma) + 0.5 * (d * q_sigma**2 + mu_norm2) / (p_sigma**2) - 0.5 * d


class PositiveLinear(torch.nn.Module):
    """Linear layer with weights forced to be positive."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 weight_init_offset: int = -2):
        super(PositiveLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(
            torch.empty((out_features, in_features)))
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        self.weight_init_offset = weight_init_offset
        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        with torch.no_grad():
            self.weight.add_(self.weight_init_offset)

        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        positive_weight = softplus(self.weight)
        return F.linear(input, positive_weight, self.bias)


class SinusoidalPosEmb(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        x = x.squeeze() * 1000
        assert len(x.shape) == 1
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
    

class PredefinedNoiseSchedule(torch.nn.Module):  # for gammas in SNR - for VP
    """
    Predefined noise schedule. Essentially creates a lookup array for predefined (non-learned) noise schedules.
    """
    def __init__(self, noise_schedule, timesteps, precision):
        super(PredefinedNoiseSchedule, self).__init__()
        self.timesteps = timesteps

        if noise_schedule == 'cosine':
            alphas2 = cosine_beta_schedule(timesteps)
        elif 'polynomial' in noise_schedule:
            splits = noise_schedule.split('_')  # e.g. polynomial_2
            assert len(splits) == 2
            power = float(splits[1])
            alphas2 = polynomial_schedule(timesteps, s=precision, power=power)
        else:
            raise ValueError(noise_schedule)

        print('alphas2', alphas2)

        sigmas2 = 1 - alphas2

        log_alphas2 = np.log(alphas2)
        log_sigmas2 = np.log(sigmas2)

        log_alphas2_to_sigmas2 = log_alphas2 - log_sigmas2

        print('gamma', -log_alphas2_to_sigmas2)

        self.gamma = torch.nn.Parameter(
            torch.from_numpy(-log_alphas2_to_sigmas2).float(),
            requires_grad=False)  # 1D

    def forward(self, t):
        t_int = torch.round(t * self.timesteps).long()
        return self.gamma[t_int]  # keeps dim - i.e. if t: [bs, 1] then out: [bs, 1]


class VENoiseSchedule(torch.nn.Module):

    def __init__(self, timesteps, rho, sigma_min, sigma_max):
        super().__init__()
        self.timesteps = timesteps
        self.rho = rho
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

        self.log_alphas = torch.nn.Parameter(  # check shapes
            torch.zeros([timesteps+1]),
            requires_grad=False
        )
        # need to use timesteps+1??? as we sample in {0, ..., T}
        self.log_sigmas2 = torch.nn.Parameter(
            torch.log(
                ve_schedule(timesteps+1, rho=rho, 
                            sigma_min=sigma_min, 
                            sigma_max=sigma_max)),
            requires_grad=False
        )

    def forward(self, t):
        t_int = torch.round(t * self.timesteps).long()
        return self.log_alphas[t_int], self.log_sigmas2[t_int]  # keeps dims to [bs, 1]


class GammaNetwork(torch.nn.Module):
    """The gamma network models a monotonic increasing function. Construction as in the VDM paper."""
    def __init__(self):
        super().__init__()

        self.l1 = PositiveLinear(1, 1)
        self.l2 = PositiveLinear(1, 1024)
        self.l3 = PositiveLinear(1024, 1)

        self.gamma_0 = torch.nn.Parameter(torch.tensor([-5.]))
        self.gamma_1 = torch.nn.Parameter(torch.tensor([10.]))
        self.show_schedule()

    def show_schedule(self, num_steps=50):
        t = torch.linspace(0, 1, num_steps).view(num_steps, 1)
        gamma = self.forward(t)
        print('Gamma schedule:')
        print(gamma.detach().cpu().numpy().reshape(num_steps))

    def gamma_tilde(self, t):
        l1_t = self.l1(t)
        return l1_t + self.l3(torch.sigmoid(self.l2(l1_t)))

    def forward(self, t):
        zeros, ones = torch.zeros_like(t), torch.ones_like(t)
        # Not super efficient.
        gamma_tilde_0 = self.gamma_tilde(zeros)
        gamma_tilde_1 = self.gamma_tilde(ones)
        gamma_tilde_t = self.gamma_tilde(t)

        # Normalize to [0, 1]
        normalized_gamma = (gamma_tilde_t - gamma_tilde_0) / (
                gamma_tilde_1 - gamma_tilde_0)

        # Rescale to [gamma_0, gamma_1]
        gamma = self.gamma_0 + (self.gamma_1 - self.gamma_0) * normalized_gamma

        return gamma


def cdf_standard_gaussian(x):
    return 0.5 * (1. + torch.erf(x / math.sqrt(2)))


class EnVariationalDiffusion(torch.nn.Module):
    """
    The E(n) Diffusion Module.
    """
    def __init__(
            self,
            dynamics: torch.nn.Module, in_node_nf: int, n_dims: int,
            timesteps: int = 1000, parametrization='eps', noise_schedule='learned',
            noise_precision=1e-4, loss_type='vlb', norm_values=(1., 1., 1.),
            norm_biases=(None, 0., 0.), include_charges=True, 
            com_free=True, rho=None, sigma_min=None, sigma_max=None, data_aug_at_sampling=False):
        super().__init__()  
        self.data_aug_at_sampling = data_aug_at_sampling
        self.use_noised_x = dynamics.use_noised_x


        # norm_values=normalize_factors [1, 4, 1], norm_biases is default - how to scale xh, vlb is default

        assert loss_type in {'vlb', 'l2'}
        self.loss_type = loss_type
        self.include_charges = include_charges
        if noise_schedule == 'learned':
            assert loss_type == 'vlb', 'A noise schedule can only be learned' \
                                       ' with a vlb objective.'

        # Only supported parametrization.
        assert parametrization == 'eps'  # experiment with this

        self.com_free = com_free
        self.sigma_max = sigma_max
        if noise_schedule == 'learned':
            self.gamma = GammaNetwork()
        else:
            if com_free:
                self.gamma = PredefinedNoiseSchedule(noise_schedule, timesteps=timesteps,
                                                     precision=noise_precision)
            else:
                self.gamma = VENoiseSchedule(timesteps, rho, sigma_min, sigma_max)

        # The network that will predict the denoising.
        self.dynamics = dynamics

        self.in_node_nf = in_node_nf  # atom decoder \pm include_charge
        self.n_dims = n_dims
        self.num_classes = self.in_node_nf - self.include_charges  # 'atom_decoder': ['H', 'C', 'N', 'O', 'F'] where does charges go

        self.T = timesteps
        self.parametrization = parametrization

        self.norm_values = norm_values
        self.norm_biases = norm_biases
        self.register_buffer('buffer', torch.zeros(1))

        if noise_schedule != 'learned':
            self.check_issues_norm_values()

    def check_issues_norm_values(self, num_stdevs=8):  # need differences in discrete values to be large compared to \sigma_0
        zeros = torch.zeros((1, 1))
        gamma_0 = self.gamma(zeros)
        sigma_0 = self.sigma(gamma_0, target_tensor=zeros).item()

        # Checked if 1 / norm_value is still larger than 10 * standard
        # deviation.
        max_norm_value = max(self.norm_values[1], self.norm_values[2])

        if sigma_0 * num_stdevs > 1. / max_norm_value:
            raise ValueError(
                f'Value for normalization value {max_norm_value} probably too '
                f'large with sigma_0 {sigma_0:.5f} and '
                f'1 / norm_value = {1. / max_norm_value}')

    def phi(self, x, t, node_mask, edge_mask, context):
        if self.com_free:
            net_out = self.dynamics._forward(t, x, node_mask, edge_mask, context)
        else:
            net_out = self.dynamics._forward(t, x, node_mask, edge_mask, context, gamma=self.gamma(t))
        return net_out

    def inflate_batch_array(self, array, target):
        """
        Inflates the batch array (array) with only a single axis (i.e. shape = (batch_size,), or possibly more empty
        axes (i.e. shape (batch_size, 1, ..., 1)) to match the target shape.
        """
        if isinstance(array, tuple):  # for dealing with com_free=False
            target_shape = (array[0].size(0),) + (1,) * (len(target.size()) - 1)  # (bs, 1, ..., 1)
            return array[0].view(target_shape), array[1].view(target_shape)
        else:
            target_shape = (array.size(0),) + (1,) * (len(target.size()) - 1)  # (bs, 1, ..., 1)
            return array.view(target_shape)

    def sigma(self, gamma, target_tensor):
        """Computes sigma given gamma."""
        if self.com_free:
            return self.inflate_batch_array(torch.sqrt(torch.sigmoid(gamma)), target_tensor)
        else:
            return self.inflate_batch_array(torch.sqrt(torch.exp(gamma[1])), target_tensor)

    def alpha(self, gamma, target_tensor):
        """Computes alpha given gamma."""
        if self.com_free:
            return self.inflate_batch_array(torch.sqrt(torch.sigmoid(-gamma)), target_tensor)  # why -gamma?
        else:
            return self.inflate_batch_array(torch.exp(gamma[0]), target_tensor)

    def SNR(self, gamma):
        """Computes signal to noise ratio (alpha^2/sigma^2) given gamma."""
        if self.com_free:
            return torch.exp(-gamma)
        else:
            return torch.exp(-gamma[1])  # as alpha=1

    def subspace_dimensionality(self, node_mask):  # [bs, n_nodes]
        """Compute the dimensionality on translation-invariant linear subspace where distributions on x are defined."""
        number_of_nodes = torch.sum(node_mask.squeeze(2), dim=1)
        if self.com_free:
            return (number_of_nodes - 1) * self.n_dims
        else:
            return number_of_nodes * self.n_dims

    def normalize(self, x, h, node_mask):
        x = x / self.norm_values[0]
        delta_log_px = -self.subspace_dimensionality(node_mask) * np.log(self.norm_values[0])

        # Casting to float in case h still has long or int type.
        h_cat = (h['categorical'].float() - self.norm_biases[1]) / self.norm_values[1] * node_mask
        h_int = (h['integer'].float() - self.norm_biases[2]) / self.norm_values[2]

        if self.include_charges:
            h_int = h_int * node_mask

        # Create new h dictionary.
        h = {'categorical': h_cat, 'integer': h_int}

        return x, h, delta_log_px

    def unnormalize(self, x, h_cat, h_int, node_mask):
        x = x * self.norm_values[0]
        h_cat = h_cat * self.norm_values[1] + self.norm_biases[1]
        h_cat = h_cat * node_mask
        h_int = h_int * self.norm_values[2] + self.norm_biases[2]

        if self.include_charges:
            h_int = h_int * node_mask

        return x, h_cat, h_int

    def unnormalize_z(self, z, node_mask):  # [bs, n_nodes, n_dims+num_classes+include_charges]
        # Parse from z
        x, h_cat = z[:, :, 0:self.n_dims], z[:, :, self.n_dims:self.n_dims+self.num_classes]
        h_int = z[:, :, self.n_dims+self.num_classes:self.n_dims+self.num_classes+1]
        assert h_int.size(2) == self.include_charges

        # Unnormalize
        x, h_cat, h_int = self.unnormalize(x, h_cat, h_int, node_mask)
        output = torch.cat([x, h_cat, h_int], dim=2)
        return output

    def sigma_and_alpha_t_given_s(self, gamma_t: torch.Tensor, gamma_s: torch.Tensor, target_tensor: torch.Tensor):
        """
        Computes sigma t given s, using gamma_t and gamma_s. Used during sampling.

        These are defined as:
            alpha t given s = alpha t / alpha s,
            sigma t given s = sqrt(1 - (alpha t given s) ^2 ).
        """
        if self.com_free:
            sigma2_t_given_s = self.inflate_batch_array(
                -expm1(softplus(gamma_s) - softplus(gamma_t)), target_tensor
            )

            # alpha_t_given_s = alpha_t / alpha_s
            log_alpha2_t = F.logsigmoid(-gamma_t)
            log_alpha2_s = F.logsigmoid(-gamma_s)
            log_alpha2_t_given_s = log_alpha2_t - log_alpha2_s

            alpha_t_given_s = torch.exp(0.5 * log_alpha2_t_given_s)
            alpha_t_given_s = self.inflate_batch_array(
                alpha_t_given_s, target_tensor)

            sigma_t_given_s = torch.sqrt(sigma2_t_given_s)

        else:
            bs = len(gamma_t[0])
            alpha_t_given_s = self.inflate_batch_array(torch.ones([bs], device=target_tensor.device), target_tensor)
            sigma2_t_given_s = self.inflate_batch_array(
                torch.exp(gamma_t[1]) - torch.exp(gamma_s[1]),
                target_tensor
            )
            sigma_t_given_s = torch.sqrt(sigma2_t_given_s)

        return sigma2_t_given_s, sigma_t_given_s, alpha_t_given_s  # [bs, 1, ..., 1]

    def kl_prior(self, xh, node_mask):
        """Computes the KL between q(z1 | x) and the prior p(z1) = Normal(0, 1).

        This is essentially a lot of work for something that is in practice negligible in the loss. However, you
        compute it so that you see it when you've made a mistake in your noise schedule.
        """
        # Compute the last alpha value, alpha_T.
        ones = torch.ones((xh.size(0), 1), device=xh.device)  # [bs, 1]
        gamma_T = self.gamma(ones)
        alpha_T = self.alpha(gamma_T, xh)  # [bs, 1, 1]

        # Compute means.
        mu_T = alpha_T * xh
        mu_T_x, mu_T_h = mu_T[:, :, :self.n_dims], mu_T[:, :, self.n_dims:]

        # Compute standard deviations (only batch axis for x-part, inflated for h-part).
        sigma_T_x = self.sigma(gamma_T, mu_T_x).squeeze()  # NOTE: Remove inflate, only keep batch dimension for x-part. - #[bs, 1]
        sigma_T_h = self.sigma(gamma_T, mu_T_h)  # [bs, 1, 1]

        if self.com_free:
            # Compute KL for h-part.
            zeros, ones = torch.zeros_like(mu_T_h), torch.ones_like(sigma_T_h)
            kl_distance_h = gaussian_KL(mu_T_h, sigma_T_h, zeros, ones, node_mask)

            # Compute KL for x-part.
            zeros, ones = torch.zeros_like(mu_T_x), torch.ones_like(sigma_T_x)
            subspace_d = self.subspace_dimensionality(node_mask)
            kl_distance_x = gaussian_KL_for_dimension(mu_T_x, sigma_T_x, zeros, ones, d=subspace_d)

        else:
            # we compare against N(0, \sigma_T^2)
            # Compute KL for h-part.
            zeros = torch.zeros_like(mu_T_h)
            kl_distance_h = gaussian_KL(mu_T_h, sigma_T_h, zeros, sigma_T_h, node_mask)

            # Compute KL for x-part.
            zeros = torch.zeros_like(mu_T_x)
            subspace_d = self.subspace_dimensionality(node_mask)
            kl_distance_x = gaussian_KL_for_dimension(mu_T_x, sigma_T_x, zeros, sigma_T_x, d=subspace_d)

        return kl_distance_x + kl_distance_h

    def compute_x_pred(self, net_out, zt, gamma_t):  # for sampling the final time-step
        """Commputes x_pred, i.e. the most likely prediction of x."""
        if self.parametrization == 'x':
            x_pred = net_out
        elif self.parametrization == 'eps':
            sigma_t = self.sigma(gamma_t, target_tensor=net_out)
            alpha_t = self.alpha(gamma_t, target_tensor=net_out)
            eps_t = net_out
            x_pred = 1. / alpha_t * (zt - sigma_t * eps_t)  # predict x_0 by [\hat{x}, \hat{h}] from z_t = \alpha_t x + \sigma_t \epsilon
        else:
            raise ValueError(self.parametrization)

        return x_pred

    def compute_error(self, net_out, gamma_t, eps):
        """Computes error, i.e. the most likely prediction of x."""
        eps_t = net_out
        if self.training and self.loss_type == 'l2':
            denom = (self.n_dims + self.in_node_nf) * eps_t.shape[1]
            error = sum_except_batch((eps - eps_t) ** 2) / denom
        else:
            error = sum_except_batch((eps - eps_t) ** 2)  # [bs, n_nodes, dim] to [bs]
        return error

    def log_constants_p_x_given_z0(self, x, node_mask):  # for L_0 - NOTE: change for our approach
        """Computes p(x|z0)."""
        batch_size = x.size(0)

        if self.com_free:
            n_nodes = node_mask.squeeze(2).sum(1)  # N has shape [B]
            assert n_nodes.size() == (batch_size,)
            degrees_of_freedom_x = (n_nodes - 1) * self.n_dims

            zeros = torch.zeros((x.size(0), 1), device=x.device)  # [bs, 1]
            gamma_0 = self.gamma(zeros)

            # Recall that sigma_x = sqrt(sigma_0^2 / alpha_0^2) = SNR(-0.5 gamma_0).
            log_sigma_x = 0.5 * gamma_0.view(batch_size)

        else:
            n_nodes = node_mask.squeeze(2).sum(1)  # N has shape [B]
            assert n_nodes.size() == (batch_size,)
            degrees_of_freedom_x = n_nodes * self.n_dims

            zeros = torch.zeros((x.size(0), 1), device=x.device)  # [bs, 1]
            gamma_0 = self.gamma(zeros)

            # Recall that sigma_x = sqrt(sigma_0^2 / alpha_0^2) = SNR(-0.5 gamma_0).
            log_sigma_x = 0.5 * gamma_0[1].view(batch_size)

        return degrees_of_freedom_x * (- log_sigma_x - 0.5 * np.log(2 * np.pi))

    def sample_p_xh_given_z0(self, z0, node_mask, edge_mask, context, fix_noise=False):  # last time step
        """Samples x ~ p(x|z0)."""
        zeros = torch.zeros(size=(z0.size(0), 1), device=z0.device)  
        gamma_0 = self.gamma(zeros)  # [bs, 1]
        # Computes sqrt(sigma_0^2 / alpha_0^2)
        if self.com_free:
            sigma_x = self.SNR(-0.5 * gamma_0).unsqueeze(1)  # [bs]
        else:
            sigma_x = torch.exp(0.5 * gamma_0[1]).unsqueeze(1)
        
        if self.data_aug_at_sampling:            
            print("Applying data augmentation at sampling time")
            # Get the matrix or matrices to use for augmentation def random_rotation(x, output_matrix=False, use_matrices=None):
            temp_z0 = deepcopy(z0)
            z0_x = temp_z0[:, :, :3]
            z0_h = temp_z0[:, :, 3:]

            # Sample matrix and apply rotation
            # matrices = data_aug_utils.random_rotation(z0_x, output_matrix=True)
            # z0_x = data_aug_utils.random_rotation(z0_x, use_matrices=matrices)
            g = sym_nn_utils.orthogonal_haar(dim=3, target_tensor=z0_x)
            z0_x = torch.bmm(z0_x, g)
            temp_z0 = torch.cat([z0_x, z0_h], dim=2)
            net_out = self.phi(temp_z0, zeros, node_mask, edge_mask, context)  # [bs, n_nodes, dims]

            # Apply the inverse rotation (orthogonal matrix) to the output
            # Get inverse by transposing
            # inverse_matrices = []
            # for matrix in matrices:
            #     inverse_matrices.append(matrix.transpose(1, 2))
            # inverse_matrices = inverse_matrices[::-1]  # reverse the order
            
            net_out_x = net_out[:, :, :3]
            net_out_h = net_out[:, :, 3:]
            # net_out_x = data_aug_utils.random_rotation(net_out_x, use_matrices=inverse_matrices).detach()
            net_out_x = torch.bmm(net_out_x, g.transpose(1, 2))
            net_out = torch.cat([net_out_x, net_out_h], dim=2)
        else:
            net_out = self.phi(z0, zeros, node_mask, edge_mask, context)  # [bs, n_nodes, dims]

        # Compute mu for p(zs | zt).
        mu_x = self.compute_x_pred(net_out, z0, gamma_0)
        xh = self.sample_normal(mu=mu_x, sigma=sigma_x, node_mask=node_mask, 
                                fix_noise=fix_noise, com_free=self.com_free)

        x = xh[:, :, :self.n_dims]

        x = utils.remove_mean_with_mask(x, node_mask)  # NOTE: LEO check this - should be fine as log_pxh_given_z0_without_constants doesn't use x directly

        h_int = z0[:, :, -1:] if self.include_charges else torch.zeros(0).to(z0.device)
        x, h_cat, h_int = self.unnormalize(x, z0[:, :, self.n_dims:-1], h_int, node_mask)  # back to original scale

        h_cat = F.one_hot(torch.argmax(h_cat, dim=2), self.num_classes) * node_mask  # why node_mask? [bs, n_nodes, num_classes] of ints
        h_int = torch.round(h_int).long() * node_mask  # [bs, n_nodes, 1]
        h = {'integer': h_int, 'categorical': h_cat}
        return x, h

    def sample_normal(self, mu, sigma, node_mask, fix_noise=False, com_free=True):  # mu: [bs, n_nodes, dim]?
        """Samples from a Normal distribution."""
        bs = 1 if fix_noise else mu.size(0)
        eps = self.sample_combined_position_feature_noise(bs, mu.size(1), node_mask, com_free=com_free)  # CoF-free for x
        return mu + sigma * eps

    def log_pxh_given_z0_without_constants(
            self, x, h, z_t, gamma_0, eps, net_out, node_mask, epsilon=1e-10):
        # Discrete properties are predicted directly from z_t.
        z_h_cat = z_t[:, :, self.n_dims:-1] if self.include_charges else z_t[:, :, self.n_dims:]
        z_h_int = z_t[:, :, -1:] if self.include_charges else torch.zeros(0).to(z_t.device)

        # Take only part over x.
        eps_x = eps[:, :, :self.n_dims]
        net_x = net_out[:, :, :self.n_dims]

        # Compute sigma_0 and rescale to the integer scale of the data. - i.e. unnormalise
        sigma_0 = self.sigma(gamma_0, target_tensor=z_t)
        sigma_0_cat = sigma_0 * self.norm_values[1]
        sigma_0_int = sigma_0 * self.norm_values[2]

        # Computes the error for the distribution N(x | 1 / alpha_0 z_0 + sigma_0/alpha_0 eps_0, sigma_0 / alpha_0),
        # the weighting in the epsilon parametrization is exactly '1'.
        # # NOTE: this is for epsilon-parametrisation, unnormalisation cancels out for this case
        log_p_x_given_z_without_constants = -0.5 * self.compute_error(net_x, gamma_0, eps_x)

        # Compute delta indicator masks. - back to int
        h_integer = torch.round(h['integer'] * self.norm_values[2] + self.norm_biases[2]).long()
        onehot = h['categorical'] * self.norm_values[1] + self.norm_biases[1]  # NOTE: why unnormalise again - shouldn't h already be unnormalised?

        estimated_h_integer = z_h_int * self.norm_values[2] + self.norm_biases[2]
        estimated_h_cat = z_h_cat * self.norm_values[1] + self.norm_biases[1]
        assert h_integer.size() == estimated_h_integer.size()

        h_integer_centered = h_integer - estimated_h_integer

        # Compute integral from -0.5 to 0.5 of the normal distribution
        # N(mean=h_integer_centered, stdev=sigma_0_int)
        # sigma_0_int handles unnormalisation
        log_ph_integer = torch.log(
            cdf_standard_gaussian((h_integer_centered + 0.5) / sigma_0_int)
            - cdf_standard_gaussian((h_integer_centered - 0.5) / sigma_0_int)
            + epsilon)
        log_ph_integer = sum_except_batch(log_ph_integer * node_mask)

        # Centered h_cat around 1, since onehot encoded.
        centered_h_cat = estimated_h_cat - 1

        # Compute integrals from 0.5 to 1.5 of the normal distribution
        # N(mean=z_h_cat, stdev=sigma_0_cat)
        log_ph_cat_proportional = torch.log(
            cdf_standard_gaussian((centered_h_cat + 0.5) / sigma_0_cat)
            - cdf_standard_gaussian((centered_h_cat - 0.5) / sigma_0_cat)
            + epsilon)

        # Normalize the distribution over the categories.
        log_Z = torch.logsumexp(log_ph_cat_proportional, dim=2, keepdim=True)
        log_probabilities = log_ph_cat_proportional - log_Z

        # Select the log_prob of the current category usign the onehot
        # representation.
        log_ph_cat = sum_except_batch(log_probabilities * onehot * node_mask)

        # Combine categorical and integer log-probabilities.
        log_p_h_given_z = log_ph_integer + log_ph_cat

        # Combine log probabilities for x and h.
        log_p_xh_given_z = log_p_x_given_z_without_constants + log_p_h_given_z

        return log_p_xh_given_z

    def compute_loss(self, x, h, node_mask, edge_mask, context, t0_always):
        """Computes an estimator for the variational lower bound, or the simple loss (MSE)."""

        # This part is about whether to include loss term 0 always 0 whether evaluating or not
        if t0_always:
            # loss_term_0 will be computed separately.
            # estimator = loss_0 + loss_t,  where t ~ U({1, ..., T})
            lowest_t = 1
        else:
            # estimator = loss_t,           where t ~ U({0, ..., T})  - False for training
            lowest_t = 0

        # Sample a timestep t - either from 0 to T, or from 1 to T
        t_int = torch.randint(
            lowest_t, self.T + 1, size=(x.size(0), 1), device=x.device).float()  # [bs, 1] from [0, T]
        s_int = t_int - 1
        t_is_zero = (t_int == 0).float()  # Important to compute log p(x | z0)  [bs, 1] of 0, 1

        # Normalize the times steps
        # Step of s will never be used, since then p(x | z0) is computed.
        s = s_int / self.T
        t = t_int / self.T

        # Compute gamma_s and gamma_t - these are for the noise schedule.
        gamma_s = self.inflate_batch_array(self.gamma(s), x)
        gamma_t = self.inflate_batch_array(self.gamma(t), x)

        # Compute alpha_t and sigma_t from gamma - these are the means and standard deviations of the normal distribution for the noise that
        # will be added to the position and feature vectors
        alpha_t = self.alpha(gamma_t, x)  # [bs, 1, 1]
        sigma_t = self.sigma(gamma_t, x)

        # Sample noise from zt ~ Normal(alpha_t x, sigma_t)
        eps = self.sample_combined_position_feature_noise(
            n_samples=x.size(0), n_nodes=x.size(1), node_mask=node_mask, com_free=self.com_free)  # [bs, n_nodes, dims] - masks out non-atom indexes

        # Concatenate x, h[integer] and h[categorical].
        xh = torch.cat([x, h['categorical'], h['integer']], dim=2)  # [bs, n_nodes, dims]
        
        # Sample noisy observation z_t given x, h for timestep t, from q(z_t | x, h)
        # Appy the COMfree trick
        z_t = alpha_t * xh + sigma_t * eps
        if self.com_free:
            diffusion_utils.assert_mean_zero_with_mask(z_t[:, :, :self.n_dims], node_mask)

        # Using our model to predict the denoised version of z_t  # TODO
        net_out = self.phi(z_t, t, node_mask, edge_mask, context)

        # Compute the error.
        error = self.compute_error(net_out, gamma_t, eps)  # [bs]

        if self.training and self.loss_type == 'l2':
            SNR_weight = torch.ones_like(error)
        else:
            # Compute weighting with SNR: (SNR(s-t) - 1) for epsilon parametrization.
            if self.com_free:
                SNR_weight = (self.SNR(gamma_s - gamma_t) - 1).squeeze(1).squeeze(1)
            else:
                #SNR_weight = (torch.exp(-gamma_s[1]+gamma_t[1]) - 1).squeeze(1).squeeze(1)
                SNR_weight = expm1(-gamma_s[1]+gamma_t[1]).squeeze(1).squeeze(1)
        assert error.size() == SNR_weight.size()
        loss_t_larger_than_zero = 0.5 * SNR_weight * error

        # The _constants_ depending on sigma_0 from the
        # cross entropy term E_q(z0 | x) [log p(x | z0)].
        neg_log_constants = -self.log_constants_p_x_given_z0(x, node_mask)  # [bs]

        # Reset constants during training with l2 loss.
        if self.training and self.loss_type == 'l2':
            neg_log_constants = torch.zeros_like(neg_log_constants)

        # The KL between q(z1 | x) and p(z1) = Normal(0, 1). Should be close to zero. NOTE: changed for CoM
        kl_prior = self.kl_prior(xh, node_mask)  # [bs]

        # Combining the terms
        if t0_always:
            loss_t = loss_t_larger_than_zero
            num_terms = self.T  # Since t=0 is not included here.
            estimator_loss_terms = num_terms * loss_t

            # Compute noise values for t = 0.
            t_zeros = torch.zeros_like(s)
            gamma_0 = self.inflate_batch_array(self.gamma(t_zeros), x)
            alpha_0 = self.alpha(gamma_0, x)
            sigma_0 = self.sigma(gamma_0, x)

            # Sample z_0 given x, h for timestep t, from q(z_t | x, h)
            eps_0 = self.sample_combined_position_feature_noise(
                n_samples=x.size(0), n_nodes=x.size(1), node_mask=node_mask, com_free=self.com_free)
            z_0 = alpha_0 * xh + sigma_0 * eps_0

            net_out = self.phi(z_0, t_zeros, node_mask, edge_mask, context)

            loss_term_0 = -self.log_pxh_given_z0_without_constants(
                x, h, z_0, gamma_0, eps_0, net_out, node_mask)

            assert kl_prior.size() == estimator_loss_terms.size()
            assert kl_prior.size() == neg_log_constants.size()
            assert kl_prior.size() == loss_term_0.size()

            loss = kl_prior + estimator_loss_terms + neg_log_constants + loss_term_0

        else:
            # Computes the L_0 term (even if gamma_t is not actually gamma_0)
            # and this will later be selected via masking.
            loss_term_0 = -self.log_pxh_given_z0_without_constants(
                x, h, z_t, gamma_t, eps, net_out, node_mask)

            t_is_not_zero = 1 - t_is_zero

            loss_t = loss_term_0 * t_is_zero.squeeze() + t_is_not_zero.squeeze() * loss_t_larger_than_zero

            # Only upweigh estimator if using the vlb objective.
            if self.training and self.loss_type == 'l2':
                estimator_loss_terms = loss_t
            else:
                num_terms = self.T + 1  # Includes t = 0.  - NOTE: why?
                estimator_loss_terms = num_terms * loss_t  # NOTE: [bs]

            assert kl_prior.size() == estimator_loss_terms.size()
            assert kl_prior.size() == neg_log_constants.size()

            loss = kl_prior + estimator_loss_terms + neg_log_constants  # from prior term, t\in[0, 1] and logZ^{-1} constants from t=0

        assert len(loss.shape) == 1, f'{loss.shape} has more than only batch dim.'

        return loss, {'t': t_int.squeeze(), 'loss_t': loss.squeeze(),
                      'error': error.squeeze()}

    def forward(self, x, h, node_mask=None, edge_mask=None, context=None):
        """
        Computes the loss (type l2 or NLL) if training. And if eval then always computes NLL."""
        # Normalize data - take into account volume change in x
        x, h, delta_log_px = self.normalize(x, h, node_mask)

        # Reset delta_log_px if not vlb objective.
        if self.training and self.loss_type == 'l2':
            delta_log_px = torch.zeros_like(delta_log_px)

        if self.training:
            # Only 1 forward pass when t0_always is False.
            loss, loss_dict = self.compute_loss(x, h, node_mask, edge_mask, context, t0_always=False)
        else:
            # Less variance in the estimator, costs two forward passes. ?
            loss, loss_dict = self.compute_loss(x, h, node_mask, edge_mask, context, t0_always=True)

        neg_log_pxh = loss

        # Correct for normalization on x.
        assert neg_log_pxh.size() == delta_log_px.size()  # np scalar
        neg_log_pxh = neg_log_pxh - delta_log_px  # add the (neg) log contribution of normalisation 

        return neg_log_pxh


    def forward_for_checking_gamma_stoch(self, x, h, node_mask=None, edge_mask=None, context=None, t):
        # Normalize data 
        x, h, delta_log_px = self.normalize(x, h, node_mask)
        out = self.compute_loss_checking_gamma_stoch(x, h, node_mask, edge_mask, context, t)
        return out


    def compute_loss_checking_gamma_stoch(self, x, h, node_mask, edge_mask, context, t):
        t_int = torch.randint(t, t+1, size=(x.size(0), 1), device=x.device).float()  # [bs, 1] from [0, T]
        s_int = t_int - 1
        t_is_zero = (t_int == 0).float()  # Important to compute log p(x | z0)  [bs, 1] of 0, 1

        # Normalize the times steps
        # Step of s will never be used, since then p(x | z0) is computed.
        s = s_int / self.T
        t = t_int / self.T

       # Compute gamma_s and gamma_t - these are for the noise schedule.
        gamma_s = self.inflate_batch_array(self.gamma(s), x)
        gamma_t = self.inflate_batch_array(self.gamma(t), x)

        # Compute alpha_t and sigma_t from gamma - these are the means and standard deviations of the normal distribution for the noise that
        # will be added to the position and feature vectors
        alpha_t = self.alpha(gamma_t, x)  # [bs, 1, 1]
        sigma_t = self.sigma(gamma_t, x)

        # Sample noise from zt ~ Normal(alpha_t x, sigma_t)
        eps = self.sample_combined_position_feature_noise(
            n_samples=x.size(0), n_nodes=x.size(1), node_mask=node_mask, com_free=self.com_free)  # [bs, n_nodes, dims] - masks out non-atom indexes

        # Concatenate x, h[integer] and h[categorical].
        xh = torch.cat([x, h['categorical'], h['integer']], dim=2)  # [bs, n_nodes, dims]

        # Sample noisy observation z_t given x, h for timestep t, from q(z_t | x, h)
        # Appy the COMfree trick
        if self.use_noised_x:
            z_t = alpha_t * xh + sigma_t * eps
        else:
            z_t = xh
        if self.com_free:
            diffusion_utils.assert_mean_zero_with_mask(z_t[:, :, :self.n_dims], node_mask)


        # Using our model to predict the denoised version of z_t 
        if self.com_free:
            out = self.dynamics._forward(t, x, node_mask, edge_mask, context)

        return out

    def sample_p_zs_given_zt(self, s, t, zt, node_mask, edge_mask, context, fix_noise=False):
        """Samples from zs ~ p(zs | zt). Only used during sampling."""
        gamma_s = self.gamma(s)
        gamma_t = self.gamma(t)

        sigma2_t_given_s, sigma_t_given_s, alpha_t_given_s = \
            self.sigma_and_alpha_t_given_s(gamma_t, gamma_s, zt)

        sigma_s = self.sigma(gamma_s, target_tensor=zt)
        sigma_t = self.sigma(gamma_t, target_tensor=zt)

        # Neural net prediction.
        if self.data_aug_at_sampling:
            print("Applying data augmentation at sampling time")
            temp_zt = deepcopy(zt)
            zt_x = temp_zt[:, :, :3]
            zt_h = temp_zt[:, :, 3:]

            # Sample matrix and apply it to the input
            # matrices = data_aug_utils.random_rotation(zt_x, output_matrix=True)    # use orthogonal_haar to get the matrix # and then do the transpose with torch.bmm(zt_x, g)  
            # zt_x = data_aug_utils.random_rotation(zt_x, use_matrices=matrices)  # here we are applying the rotation matrix
            g = sym_nn_utils.orthogonal_haar(dim=3, target_tensor=zt_x)
            zt_x = torch.bmm(zt_x, g)  
            temp_zt = torch.cat([zt_x, zt_h], dim=2)
            eps_t = self.phi(temp_zt, t, node_mask, edge_mask, context)

            # Apply the inverse rotation (orthogonal matrix) to the output
            # Get inverse by transposing
            # inverse_matrices = []
            # for matrix in matrices:
            #     inverse_matrices.append(matrix.transpose(1, 2))
            # inverse_matrices = inverse_matrices[::-1]  # reverse the order

            eps_t_x = eps_t[:, :, :3]
            eps_t_h = eps_t[:, :, 3:]
            # eps_t_x = data_aug_utils.random_rotation(eps_t_x, use_matrices=inverse_matrices).detach()
            eps_t_x = torch.bmm(eps_t_x, g.transpose(1, 2))
            eps_t = torch.cat([eps_t_x, eps_t_h], dim=2)
        else:
            eps_t = self.phi(zt, t, node_mask, edge_mask, context)

        # Compute mu for p(zs | zt).
        if self.com_free:
            diffusion_utils.assert_mean_zero_with_mask(zt[:, :, :self.n_dims], node_mask)
            diffusion_utils.assert_mean_zero_with_mask(eps_t[:, :, :self.n_dims], node_mask)

        mu = zt / alpha_t_given_s - (sigma2_t_given_s / alpha_t_given_s / sigma_t) * eps_t

        # Compute sigma for p(zs | zt).
        sigma = sigma_t_given_s * sigma_s / sigma_t

        # Sample zs given the paramters derived from zt.
        zs = self.sample_normal(mu, sigma, node_mask, fix_noise, com_free=self.com_free)

        if self.com_free:
            # Project down to avoid numerical runaway of the center of gravity.
            zs = torch.cat(
                [diffusion_utils.remove_mean_with_mask(zs[:, :, :self.n_dims],
                                                       node_mask),
                zs[:, :, self.n_dims:]], dim=2
            )

        return zs

    def sample_combined_position_feature_noise(self, n_samples, n_nodes, node_mask, com_free=True):
        """
        Samples mean-centered normal noise for z_x, and standard normal noise for z_h.
        """
        if com_free:
            z_x = utils.sample_center_gravity_zero_gaussian_with_mask(
                size=(n_samples, n_nodes, self.n_dims), device=node_mask.device,
                node_mask=node_mask)
            z_h = utils.sample_gaussian_with_mask(
                size=(n_samples, n_nodes, self.in_node_nf), device=node_mask.device,
                node_mask=node_mask)
            z = torch.cat([z_x, z_h], dim=2)
        else:
            z = utils.sample_gaussian_with_mask(
                size=(n_samples, n_nodes, self.n_dims+self.in_node_nf), device=node_mask.device,
                node_mask=node_mask)
        return z

    @torch.no_grad()
    def sample(self, n_samples, n_nodes, node_mask, edge_mask, context, fix_noise=False):
        """
        Draw samples from the generative model.
        """
        if fix_noise:
            # Noise is broadcasted over the batch axis, useful for visualizations.
            z = self.sample_combined_position_feature_noise(1, n_nodes, node_mask, com_free=self.com_free)
        else:
            z = self.sample_combined_position_feature_noise(n_samples, n_nodes, node_mask, com_free=self.com_free)

        if self.com_free:
            diffusion_utils.assert_mean_zero_with_mask(z[:, :, :self.n_dims], node_mask)
        else:
            z = self.sigma_max * z

        # Iteratively sample p(z_s | z_t) for t = 1, ..., T, with s = t - 1.
        for s in reversed(range(0, self.T)):
            s_array = torch.full((n_samples, 1), fill_value=s, device=z.device)
            t_array = s_array + 1
            s_array = s_array / self.T
            t_array = t_array / self.T

            z = self.sample_p_zs_given_zt(s_array, t_array, z, node_mask, edge_mask, context, fix_noise=fix_noise)

        # Finally sample p(x, h | z_0).
        x, h = self.sample_p_xh_given_z0(z, node_mask, edge_mask, context, fix_noise=fix_noise)  # we remove the mean even for com_free=False

        diffusion_utils.assert_mean_zero_with_mask(x, node_mask)

        max_cog = torch.sum(x, dim=1, keepdim=True).abs().max().item()
        if max_cog > 5e-2:
            print(f'Warning cog drift with error {max_cog:.3f}. Projecting '
                  f'the positions down.')
            x = diffusion_utils.remove_mean_with_mask(x, node_mask)

        return x, h

    @torch.no_grad()
    def sample_chain(self, n_samples, n_nodes, node_mask, edge_mask, context, keep_frames=None):
        """
        Draw samples from the generative model, keep the intermediate states for visualization purposes.
        """
        z = self.sample_combined_position_feature_noise(n_samples, n_nodes, node_mask, com_free=self.com_free)

        if self.com_free:
            diffusion_utils.assert_mean_zero_with_mask(z[:, :, :self.n_dims], node_mask)
        else:
            z = self.sigma_max * z

        if keep_frames is None:
            keep_frames = self.T
        else:
            assert keep_frames <= self.T
        chain = torch.zeros((keep_frames,) + z.size(), device=z.device)

        # Iteratively sample p(z_s | z_t) for t = 1, ..., T, with s = t - 1.
        for s in reversed(range(0, self.T)):
            s_array = torch.full((n_samples, 1), fill_value=s, device=z.device)
            t_array = s_array + 1
            s_array = s_array / self.T
            t_array = t_array / self.T

            z = self.sample_p_zs_given_zt(
                s_array, t_array, z, node_mask, edge_mask, context)

            if self.com_free:
                diffusion_utils.assert_mean_zero_with_mask(z[:, :, :self.n_dims], node_mask)

            # Write to chain tensor.
            write_index = (s * keep_frames) // self.T
            chain[write_index] = self.unnormalize_z(z, node_mask)

        # Finally sample p(x, h | z_0).
        x, h = self.sample_p_xh_given_z0(z, node_mask, edge_mask, context)  # we remove the mean even for com_free=False

        diffusion_utils.assert_mean_zero_with_mask(x[:, :, :self.n_dims], node_mask)

        xh = torch.cat([x, h['categorical'], h['integer']], dim=2)
        chain[0] = xh  # Overwrite last frame with the resulting x and h.

        chain_flat = chain.view(n_samples * keep_frames, *z.size()[1:])

        return chain_flat

    def log_info(self):
        """
        Some info logging of the model.
        """
        if self.com_free:
            gamma_0 = self.gamma(torch.zeros(1, device=self.buffer.device))
            gamma_1 = self.gamma(torch.ones(1, device=self.buffer.device))

            log_SNR_max = -gamma_0
            log_SNR_min = -gamma_1

            info = {
                'log_SNR_max': log_SNR_max.item(),
                'log_SNR_min': log_SNR_min.item()}
        else:
            info = "Not using CoM-free systems!"

        print(info)    
        return info
