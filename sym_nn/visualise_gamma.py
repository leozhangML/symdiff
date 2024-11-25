import torch
import numpy as np
import matplotlib.pyplot as plt
import datetime
import os
from itertools import combinations
from sklearn.decomposition import PCA

from equivariant_diffusion.en_diffusion import EnVariationalDiffusion
from equivariant_diffusion.utils import assert_mean_zero_with_mask, remove_mean_with_mask, \
     assert_correctly_masked, sample_center_gravity_zero_gaussian_with_mask
from equivariant_diffusion import utils as diffusion_utils

from sym_nn.utils import orthogonal_haar, qr

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

def create_interactive_plot(embeddings, title=""):
    """
    Creates multiple views of the same 3D embedding for better visualization.
    
    Args:
        embeddings: numpy array of shape (n_samples, 3)
        labels: numpy array of shape (n_samples,) containing class labels
    """
    # Create a figure with multiple views
    fig = plt.figure(figsize=(20, 5))
    fig.suptitle(title)

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
            alpha=0.2,
            s=10
        )

        # Customize subplot
        ax.set_title(name)
        ax.view_init(elev, azim)
        if i == 1:  # Only add legend to first plot
            ax.legend(bbox_to_anchor=(1.15, 1), loc='upper left')
    
    plt.tight_layout()
    return fig

def create_single_view_plot(embeddings, title=""):
    fig = plt.figure(figsize=(20, 5))
    fig.suptitle(title)

    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.scatter(
            embeddings[:, 0],
            embeddings[:, 1],
            embeddings[:, 2],
            alpha=0.2,
            s=10
        )

    plt.tight_layout()
    return fig

def create_hist2d_plot(coords, title=""):
    """
    Creates 2D histograms of different projections.

    Args:
        coords: np.array of shape (n_samples, 3)
        title: string for the figure title
    """
    # create figure
    fig = plt.figure(figsize=(20, 5))
    fig.suptitle(title)

    # different projections
    angle_names = ["phi", "theta", "psi"]
    idxs = [0, 1, 2]

    for i, (idx1, idx2) in enumerate(combinations(idxs, 2), 1):
        ax = fig.add_subplot(1, 3, i)
        ax.hist2d(coords[:, idx1], coords[:, idx2], bins=100)
        ax.set_title(f"{angle_names[idx1]}/{angle_names[idx2]}")

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

@torch.no_grad()
def extract_gamma_enc(args, generative_model, t, x, node_mask, use_haar=True, ignore_x=False, use_haar_out=False):
    """
    Plots a histogram of the norms and cosine metric of the gamma encoder, as well as
    the PCA components of gamma encoder.
    """
    bs, n_nodes, _ = x.shape

    print(f"use_haar={use_haar}")
    print(f"ignore_x={ignore_x}")

    if use_haar:
        g = orthogonal_haar(dim=generative_model.dynamics.n_dims, target_tensor=x)  # [bs, 3, 3]
        #g = generative_model.dynamics.base_gamma(t, x, node_mask)  # [bs, 3, 3]
    else:
        g = torch.eye(3, device=x.device)[None, ...].repeat_interleave(bs, dim=0)

    N = torch.sum(node_mask, dim=1, keepdims=True)  # [bs, 1, 1]
    pos_emb_test = generative_model.dynamics.gaussian_embedder_test(x.clone(), node_mask)  # [bs, n_nodes, n_nodes, K]
    pos_emb_test = torch.sum(generative_model.dynamics.pos_embedder_test(pos_emb_test), dim=-2) / N  # [bs, n_nodes, pos_embedder_test]

    g_inv_x = torch.bmm(x.clone(), g.clone())  # as x is represented row-wise

    #g_inv_x = node_mask * generative_model.dynamics.gamma_input_layer(g_inv_x)
    #g_inv_x = node_mask * generative_model.dynamics.gamma_projection(g_inv_x)

    if generative_model.dynamics.noise_dims > 0:
        print("USING NOISE DIMS > 0!")

        g_inv_x = torch.cat([
            g_inv_x, 
            node_mask * generative_model.dynamics.noise_std * torch.randn(
                bs, n_nodes, generative_model.dynamics.noise_dims, device=generative_model.dynamics.device
                )
            ], dim=-1)

        # remove noise
        """
        g_inv_x = torch.cat([
            g_inv_x, 
            node_mask * generative_model.dynamics.noise_std * torch.zeros(
                bs, n_nodes, generative_model.dynamics.noise_dims, device=generative_model.dynamics.device
                )
            ], dim=-1)
       """ 

    if ignore_x:
        g_inv_x = torch.zeros_like(g_inv_x, device=x.device)

    #g_inv_x = node_mask * generative_model.dynamics.gamma_input_layer(g_inv_x)

    g_inv_x = torch.cat([g_inv_x, pos_emb_test.clone()], dim=-1)

    # [bs, n_nodes, hidden_size]
    gamma = node_mask * generative_model.dynamics.gamma_enc(
        g_inv_x, t.squeeze(-1), node_mask.squeeze(-1), 
        use_final_layer=False
        )

    # extract gamma representations
    gamma = torch.sum(gamma, dim=1) / N.squeeze(-1)  # [bs, hidden_size]

    gamma_dec = generative_model.dynamics.gamma_dec(gamma)  # [bs, 9]

    gamma_mat = qr(gamma_dec.reshape(-1, generative_model.n_dims, generative_model.n_dims))[0]
    gamma_mat_coords = convert_gamma_to_coords(gamma_mat.detach().cpu())

    if use_haar_out:
        #g = orthogonal_haar(dim=generative_model.dynamics.n_dims, target_tensor=x)
        g = generative_model.dynamics.base_gamma(t, x, node_mask)  # [bs, 3, 3]
    gamma_out = torch.bmm(gamma_mat, g.transpose(2, 1))
    gamma_out_coords = convert_gamma_to_coords(gamma_out.detach().cpu())

    # check gamma values
    print("GAMMA ENCODER")
    print(gamma[0])
    print(gamma[-1])

    print("GAMMA DECODER")
    print(gamma_dec[0])
    print(gamma_dec[-1])

    # init cosine metric and PCA
    cos = torch.nn.CosineSimilarity(dim=-1)
    pca = PCA(n_components=2)

    # plot metrics for gamma encoder
    fig, axes = plt.subplots(3, 1, figsize=(20, 15))
    fig.suptitle(f"gamma encoder (FIX NOISE): , use_haar={use_haar}, ignore_x={ignore_x}")

    # gamma norms
    gamma_norms = torch.norm(gamma, dim=-1)  # [bs]
    axes[0].hist(gamma_norms.detach().cpu(), bins=100)
    axes[0].set_title("gamma encoder norms")

    # gamma cosine
    ones = torch.ones_like(gamma, device=gamma.device)
    cosine_metric = cos(ones, gamma)
    axes[1].hist(cosine_metric.detach().cpu(), bins=100)
    axes[1].set_title("cosine metric encoder wrt ones")

    # gamma PCA
    pca.fit(gamma.detach().cpu().numpy())
    pca_x = pca.transform(gamma.detach().cpu().numpy())
    axes[2].scatter(pca_x[:, 0], pca_x[:, 1], alpha=0.3)
    axes[2].set_title(f"pca of gamma encoder: explained_variance_ratio={pca.explained_variance_ratio_}")

    save_fig(args, fig)

    # plot metrics for gamma decoder
    fig, axes = plt.subplots(3, 1, figsize=(20, 15))
    fig.suptitle(f"gamma decoder: t={t[0].item()}, use_haar={use_haar}, ignore_x={ignore_x}")

    gamma_dec_norms = torch.norm(gamma_dec, dim=-1)
    axes[0].hist(gamma_dec_norms.detach().cpu(), bins=100)
    axes[0].set_title("gamma decoder norms")

    ones = torch.ones_like(gamma_dec, device=gamma_dec.device)
    cosine_metric_dec = cos(ones, gamma_dec)
    axes[1].hist(cosine_metric_dec.detach().cpu(), bins=100)
    axes[1].set_title("cosine metric decoder wrt ones")

    pca.fit(gamma_dec.detach().cpu().numpy())
    pca_x = pca.transform(gamma_dec.detach().cpu().numpy())
    axes[2].scatter(pca_x[:, 0], pca_x[:, 1], alpha=0.3)
    axes[2].set_title(f"pca of gamma decoder: explained_variance_ratio={pca.explained_variance_ratio_}")

    save_fig(args, fig)

    # plot euler angles for gamma mat
    fig = create_interactive_plot(
        gamma_mat_coords, title=f"gamma mat euler angles: t={t[0].item()}, use_haar={use_haar}, ignore_x={ignore_x}")
    save_fig(args, fig)

    # plot euler angles for gamma out
    fig = create_interactive_plot(
        gamma_out_coords, title=f"gamma out euler angles: t={t[0].item()}, use_haar={use_haar}, ignore_x={ignore_x}, use_haar_out={use_haar_out}")
    save_fig(args, fig)

    # look at base gamma
    g = generative_model.dynamics.base_gamma(t, x, node_mask)
    base_gamma_coords = convert_gamma_to_coords(g.detach().cpu())
    fig = create_interactive_plot(
        base_gamma_coords, title=f"base gamma euler angles: t={t[0].item()}")
    save_fig(args, fig)


@torch.no_grad()
def sample_gamma(args, eval_args, generative_model, z_t, t, node_mask, num_samples):
    # z_t: [1, n_nodes, n_dims+in_nodes_nf] etc.

    # repeat inputs
    z_ts = z_t.repeat_interleave(num_samples, dim=0)
    x_ts = z_ts[..., :3]
    ts = t.repeat_interleave(num_samples, dim=0)
    node_masks = node_mask.repeat_interleave(num_samples, dim=0)

    # sample multiple gammas
    gammas = generative_model.dynamics.gamma(ts, x_ts, node_masks)

    # extract euler angles
    coords = convert_gamma_to_coords(gammas.detach().cpu())

    print(np.sum(np.isnan(coords)))

    # create folder for saving plots
    fig_folder_datatime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fig_folder_name = "_".join([fig_folder_datatime, args.exp_name, args.dataset])
    args.fig_folder_path = os.path.join(eval_args.plots_path, fig_folder_name)
    try:
        os.makedirs(args.fig_folder_path)
    except OSError:
        pass

    # plot 3d scatter plot of gamma
    #fig = create_interactive_plot(coords, title=f"t={t.item()}")
    #save_fig(args, fig)

    fig = create_single_view_plot(coords, title="SymDiff")
    save_fig(args, fig)

    # plot projection heatmaps of gamma
    fig = create_hist2d_plot(coords, title=f"t={t.item()}")
    save_fig(args, fig)

    # gamma
    extract_gamma_enc(
        args, generative_model, ts[:1000], 
        x_ts[:1000], node_masks[:1000]
        )

    # gamma without haar
    extract_gamma_enc(
        args, generative_model, ts[:1000], 
        x_ts[:1000], node_masks[:1000], use_haar=False
        )

    # gamma when ignoring x
    extract_gamma_enc(
        args, generative_model, ts[:1000], 
        x_ts[:1000], node_masks[:1000], use_haar=False, 
        ignore_x=True
        )

    # gamma_out with independent haar
    extract_gamma_enc(
        args, generative_model, ts[:1000], 
        x_ts[:1000], node_masks[:1000], use_haar_out=True
        )
