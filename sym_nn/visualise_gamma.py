import torch
import numpy as np
import matplotlib.pyplot as plt
import datetime
import os

from equivariant_diffusion.en_diffusion import EnVariationalDiffusion
from equivariant_diffusion.utils import assert_mean_zero_with_mask, remove_mean_with_mask, \
     assert_correctly_masked, sample_center_gravity_zero_gaussian_with_mask
from equivariant_diffusion import utils as diffusion_utils

from sym_nn.utils import orthogonal_haar

def rotation_matrix_to_euler_angles(R):
    """
    Convert a 3D rotation matrix to Euler angles (phi, theta, psi).
    
    Args:
        R (numpy.ndarray): A 3x3 rotation matrix.
        
    Returns:
        tuple: Euler angles (phi, theta, psi) in radians.
    """
    # Extract elements from the rotation matrix
    R11, R12, R13 = R[0]
    R21, R22, R23 = R[1]
    R31, R32, R33 = R[2]
    
    # Check for the case when R31 is ±1 (gimbal lock)
    if R31 != 1 and R31 != -1:
        # General case
        theta1 = -np.arcsin(R31)
        theta2 = np.pi - theta1

        # Calculate psi and phi for both the theta1 and theta2 cases
        psi1 = np.arctan2(R32 / np.cos(theta1), R33 / np.cos(theta1))
        psi2 = np.arctan2(R32 / np.cos(theta2), R33 / np.cos(theta2))

        phi1 = np.arctan2(R21 / np.cos(theta1), R11 / np.cos(theta1))
        phi2 = np.arctan2(R21 / np.cos(theta2), R11 / np.cos(theta2))

        return (phi1, theta1, psi1), (phi2, theta2, psi2)
    
    else:
        # Special case for gimbal lock (R31 = ±1)
        if R31 == -1:
            theta = np.pi / 2
            psi = np.arctan2(R12, R13)
            phi = 0  # Can set to any value, conventionally 0
        elif R31 == 1:
            theta = -np.pi / 2
            #psi = -np.arctan2(-R12, -R13)
            psi = np.arctan2(-R12, -R13)  # from https://eecs.qmul.ac.uk/~gslabaugh/publications/euler.pdf
            phi = 0  # Can set to any value, conventionally 0

def convert_gamma_to_coords(gamma: torch.Tensor):
    # gamma: [N, 3, 3]
    euler_angles = [
        torch.stack(rotation_matrix_to_euler_angles(gamma_mat)[0]) 
        for gamma_mat in gamma
    ]
    return torch.stack(euler_angles, dim=0).numpy()

def create_interactive_plot(embeddings):
    """
    Creates multiple views of the same 3D embedding for better visualization.
    
    Args:
        embeddings: numpy array of shape (n_samples, 3)
        labels: numpy array of shape (n_samples,) containing class labels
    """
    # Create a figure with multiple views
    fig = plt.figure(figsize=(20, 5))
    
    # Different viewing angles
    views = [
        (30, 45),    # Standard view
        (0, 0),      # Front view
        (0, 90),     # Side view
        (90, 0)      # Top view
    ]
    view_names = ['Standard View', 'Front View', 'Side View', 'Top View']
    
    for i, ((elev, azim), name) in enumerate(zip(views, view_names), 1):
        ax = fig.add_subplot(1, 4, i, projection='3d')
        
        ax.scatter(
            embeddings[:, 0],
            embeddings[:, 1],
            embeddings[:, 2],
            alpha=0.3,
            s=30
        )
        
        # Customize subplot
        ax.set_title(name)
        ax.view_init(elev, azim)
        if i == 1:  # Only add legend to first plot
            ax.legend(bbox_to_anchor=(1.15, 1), loc='upper left')
    
    plt.tight_layout()
    return fig

def check_mask_correct(variables, node_mask):
    for i, variable in enumerate(variables):
        if len(variable) > 0:
            assert_correctly_masked(variable, node_mask)

def get_samples(args, generative_model, dataloader):

    device = args.device  # check
    dtype = args.dtype

    # load data
    data = next(iter(dataloader))

    x = data['positions'].to(device, dtype)
    node_mask = data['atom_mask'].to(device, dtype).unsqueeze(2)  # [bs, n_nodes, 1]
    edge_mask = data['edge_mask'].to(device, dtype)  # [bs*n_nodes^2, 1]
    one_hot = data['one_hot'].to(device, dtype)  # [bs, n_nodes, num_classes - i.e. 5 for qm9]
    charges = (data['charges'] if args.include_charges else torch.zeros(0)).to(device, dtype)  # [bs, n_nodes, 1], torch.zeros(0) for 2D experiments

    x = remove_mean_with_mask(x, node_mask)

    check_mask_correct([x, one_hot, charges], node_mask) if args.molecule else None
    assert_mean_zero_with_mask(x, node_mask)

    h = {'categorical': one_hot, 'integer': charges}

    # normalise data
    x, h, _ = generative_model.normalize(x, h, node_mask)

    # sample noised data
    lowest_t = 1  # we don't sample at t=0
    t_int = torch.randint(
            lowest_t, generative_model.T + 1, size=(x.size(0), 1), device=x.device).float()  # [bs, 1] from [0, T]
    s_int = t_int - 1
    t_is_zero = (t_int == 0).float()  # Important to compute log p(x | z0)  [bs, 1] of 0, 1

    # Normalize t to [0, 1]. Note that the negative
    # step of s will never be used, since then p(x | z0) is computed.
    s = s_int / generative_model.T
    t = t_int / generative_model.T

    # Compute gamma_s and gamma_t via the network.
    gamma_s = generative_model.inflate_batch_array(generative_model.gamma(s), x)
    gamma_t = generative_model.inflate_batch_array(generative_model.gamma(t), x)

    # Compute alpha_t and sigma_t from gamma.
    alpha_t = generative_model.alpha(gamma_t, x)  # [bs, 1, 1]
    sigma_t = generative_model.sigma(gamma_t, x)

    # Sample zt ~ Normal(alpha_t x, sigma_t)
    eps = generative_model.sample_combined_position_feature_noise(
        n_samples=x.size(0), n_nodes=x.size(1), node_mask=node_mask, com_free=generative_model.com_free)  # [bs, n_nodes, dims] - masks out non-atom indexes

    # Concatenate x, h[integer] and h[categorical].
    xh = torch.cat([x, h['categorical'], h['integer']], dim=2)  # [bs, n_nodes, dims]
    # Sample z_t given x, h for timestep t, from q(z_t | x, h)
    z_t = alpha_t * xh + sigma_t * eps

    if generative_model.com_free:
        diffusion_utils.assert_mean_zero_with_mask(z_t[:, :, :generative_model.n_dims], node_mask)

    return z_t[[0]], t[[0]], node_mask[[0]]

def save_fig(args, fig):
    dt = str(datetime.datetime.now())  # for saving plots etc.
    fig_name = "_".join([dt, args.exp_name, args.dataset])
    fig_name = fig_name + ".png"
    fig_path = os.path.join(
        args.fig_folder_path, fig_name
    )
    fig.savefig(fig_path)
    plt.close()

def sample_gamma(args, eval_args, generative_model, z_t, t, node_mask, num_samples):
    # above are just one sample

    z_ts = z_t.repeat_interleave(num_samples, dim=0)
    ts = t.repeat_interleave(num_samples, dim=0)
    node_masks = node_mask.repeat_interleave(num_samples, dim=0)

    gammas = generative_model.dynamics.gamma(ts, z_ts, node_masks)

    coords = convert_gamma_to_coords(gammas)

    # create folder for saving plots
    fig_folder_datatime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fig_folder_name = "_".join([fig_folder_datatime, args.exp_name, args.dataset])
    args.fig_folder_path = os.path.join(eval_args.plots_path, fig_folder_name)
    try:
        os.makedirs(args.fig_folder_path)
    except OSError:
        pass

    fig = create_interactive_plot(gammas)
    save_fig(args, fig)