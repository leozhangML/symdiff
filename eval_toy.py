import argparse
import os
import pickle
import datetime

import torch
import numpy as np
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt

from qm9.models import get_model
from train_test import test
from equivariant_diffusion.utils import assert_mean_zero_with_mask, remove_mean_with_mask, assert_correctly_masked

from sym_nn.distributions import retrieve_dataloaders
from sym_nn.utils import orthogonal_haar

import warnings
warnings.filterwarnings("ignore")


def save_fig(args, fig):
    dt = str(datetime.datetime.now())  # for saving plots etc.
    fig_name = "_".join([dt, args.exp_name, args.dataset])
    fig_name = fig_name + ".png"
    fig_path = os.path.join(
        args.fig_folder_path, fig_name
    )
    fig.savefig(fig_path)
    plt.close()


def convert_cpu_detach(*args):
    np_args = (arg.cpu().detach() for arg in args)
    if len(args) == 1:
        return next(np_args)
    else:
        return np_args


def plot_positions(args, positions, project=False, plot_single=False):
    # positions: [n, 2, 2] (np.array)
    fig, axes = plt.subplots(1, 1)
    if project:
        radii = np.linalg.norm(positions, axis=-1)[:, 0]  # [n]
        _ = axes.hist(radii, bins=100)
    else:
        axes.scatter(positions[:, 0, 0], positions[:, 0, 1], alpha=0.5)
        if not plot_single:
            axes.scatter(positions[:, 1, 0], positions[:, 1, 1], alpha=0.5)
        axes.set_aspect('equal')

    fig.suptitle(f"{args.exp_name}\nplot_positions")
    save_fig(args, fig)


def plot_positions_1d(args, positions):
    # positions: [n, 2, 2] (np.array)
    fig, axes = plt.subplots(1, 1)
    radii = np.linalg.norm(positions, axis=-1)[:, 0]  # [n]
    axes.scatter(radii, np.zeros(len(radii)), alpha=0.5)

    fig.suptitle(f"{args.exp_name}\nplot_positions_1d")
    save_fig(args, fig)


def plot_radius(args, positions, plot_diff=False):
    radius_0 = np.linalg.norm(positions[:, 0], axis=-1)  # [n]
    radius_1 = np.linalg.norm(positions[:, 1], axis=-1)  # [n]
    fig, axes = plt.subplots(1, 1)
    if plot_diff:
        radius_diff = np.sqrt((radius_0 - radius_1)**2)
        _ = axes.hist(radius_diff, bins=100)
    else:
        _ = axes.hist(radius_0, bins=100)
    fig.suptitle(f"{args.exp_name}\nplot_radius")
    save_fig(args, fig)


def sample(eval_args, generative_model, fix_noise=False, return_np=False):

    node_mask = torch.ones(eval_args.n_samples, 2, 1, 
                           device=generative_model.dynamics.device)
    x, _ = generative_model.sample(eval_args.n_samples, 2, node_mask, 
                                   None, None, fix_noise=fix_noise)
    if return_np:
        x = x.cpu().detach().numpy()  # [n_samples, 2, 2]

    return x


def sample_chain(eval_args, generative_model):

    node_mask = torch.ones(eval_args.n_samples, 2, 1, 
                           device=generative_model.dynamics.device)
    chain = generative_model.sample_chain(eval_args.n_samples, 2, node_mask, 
                                         None, None)
    chain_x = chain[:, :, :2].cpu().detach().numpy()  # [n_samples * timesteps, 2, 2]

    return chain_x


def sample_multiple_outputs(args, xh, t, generative_model, n=5, output_type="model", return_h=False):
    # xh: [1, 2, 3] or [1, 2, 2]
    # t: [1]

    x = xh[:, :, :2].repeat_interleave(n, dim=0)  # [n, 2, 2]
    xh = xh.repeat_interleave(n, dim=0)  # [n, 2, 3]

    t = t.repeat_interleave(n)
    node_mask = torch.ones(n, 2, 1, device=args.device)

    if output_type == "model":
        xh, gamma = generative_model.dynamics._forward(
            t, xh, node_mask, None, None, return_gamma=True
        )
        if return_h:
            return xh, gamma
        else:
            return xh[:, :, :2], gamma

    elif output_type == "gamma":
        gamma = generative_model.dynamics.gamma(t, x, node_mask)
        return gamma

    elif output_type == "k":
        xh = generative_model.dynamics.k(
            t, x, torch.zeros(n, 2, 1, device=args.device), 
            node_mask)
        if return_h:
            return xh
        else:
            return xh[:, :, :2]

    else:
        raise ValueError


def plot_outputs(args, xh, t, generative_model, n=5, plot_gamma=True):
    # xh: [1, 2, 3]
    # t: [1]

    x_0 = xh[0, :, :2]  # [2, 2]

    if plot_gamma:
        # gammas look consistent for less_noise
        xh, gamma = sample_multiple_outputs(
            args, xh, t, generative_model, n=n, output_type="model")
        angles = gamma[:, :, 0]  # from action on (1, 0) on [n, 2, 2]
    else:
        xh = sample_multiple_outputs(
            args, xh, t, generative_model, n=n, output_type="k")

    x = xh[:, :, :2]  # [n, 2, 2]

    print(f"x: {x_0}")
    print(f"t: {t}")

    x_0, x = convert_cpu_detach(x_0, x)

    fig, axes = plt.subplots(1, 1)

    axes.scatter(x_0[0, 0], x_0[0, 1], label="x0_1", c="black")
    axes.scatter(x_0[1, 0], x_0[1, 1], label="x0_2", c="yellow")

    axes.scatter(x[:, 0, 0], x[:, 0, 1], alpha=0.5, label="x_1")
    axes.scatter(x[:, 1, 0], x[:, 1, 1], alpha=0.5, label="x_2")

    if plot_gamma:
        angles = convert_cpu_detach(angles)
        axes.scatter(angles[:, 0], angles[:, 1], alpha=0.5, label="gamma")

    fig.suptitle(f"{args.exp_name}\nplot_outputs: t={t[0].item()}, plot_gamma={plot_gamma}")
    axes.legend()
    axes.set_aspect('equal')
    save_fig(args, fig)

    if plot_gamma:
        return x, gamma
    else:
        return x


def compute_angles(gamma):
    # gamma: [n, 2, 2]
    vector = gamma[:, :, 0]  # [n, 2]
    neg_y = vector[:, 1] < 0  # [n]
    
    norm = torch.norm(vector, dim=-1)  # [n]
    angles = torch.arccos(vector[:, 0] / norm)  # in [0, \pi]
    angles[neg_y] = (2 * torch.pi) - angles[neg_y]

    #angles = torch.arctan(angles[:, 1] / angles[:, 0])
    #angles = angles % (2 * torch.pi)
    return angles


def plot_gamma_hist(args, iter_dataloader, generative_model, n=5):
    # xh: [1, 2, 3]
    # t: [1]

    x, t = get_x_t(args, iter_dataloader, generative_model, add_noise=True)
    xh = convert_x_to_xh(x)

    xh0 = xh * 1.0

    # [n, 2, 2]
    _, gamma_model = sample_multiple_outputs(args, xh, t, generative_model, n=n, output_type="model")
    gamma_gamma = sample_multiple_outputs(args, xh, t, generative_model, n=n, output_type="gamma")

    angles_model = compute_angles(gamma_model)
    angles_gamma = compute_angles(gamma_gamma)
    angles_model, angles_gamma = convert_cpu_detach(angles_model, angles_gamma)

    print(torch.norm(xh - xh0))

    fig, axes = plt.subplots(1, 1, figsize=(15, 8))
    _ = axes.hist(angles_model, bins=100, alpha=0.5, label="model")
    _ = axes.hist(angles_gamma, bins=100, alpha=0.5, label="gamma")
    fig.suptitle(f"{args.exp_name}\nplot_gamma_hist: n={n}")
    save_fig(args, fig)


def check_stoc_equivariance_gamma_hist(args, iter_dataloader, generative_model, n=100, use_model=True):

    x, t = get_x_t(args, iter_dataloader, generative_model, add_noise=True)
    output_type = "model" if use_model else "gamma"

    # [1, 2, 2]
    g = orthogonal_haar(
        dim=2, target_tensor=torch.empty(1, device=args.device)
    )

    # [1, 2, 3]
    xh = convert_x_to_xh(x)
    g_xh = convert_x_to_xh(torch.bmm(x, g.transpose(1, 2)))

    # [n, 2, 2]
    gamma_x = sample_multiple_outputs(args, xh, t, generative_model, n=n, output_type=output_type)
    gamma_x = gamma_x[-1] if use_model else gamma_x

    gamma_g_x = sample_multiple_outputs(args, g_xh, t, generative_model, n=n, output_type=output_type)
    gamma_g_x = gamma_g_x[-1] if use_model else gamma_g_x

    print("gamma_x: ", gamma_x.shape)
    print("g: ", g.shape)

    g_gamma_x = torch.bmm(g.repeat_interleave(n, dim=0), gamma_x)

    angles_gamma_x = compute_angles(gamma_x)
    angles_gamma_g_x = compute_angles(gamma_g_x)
    angles_g_gamma_x = compute_angles(g_gamma_x)

    angles_gamma_x, angles_gamma_g_x, angles_g_gamma_x = convert_cpu_detach(
        angles_gamma_x, angles_gamma_g_x, angles_g_gamma_x)

    fig, axes = plt.subplots(1, 3, figsize=(15, 8))
    _ = axes[0].hist(angles_gamma_x, bins=100)
    _ = axes[1].hist(angles_gamma_g_x, bins=100)
    _ = axes[2].hist(angles_g_gamma_x, bins=100)

    axes[0].set_title("angles_gamma_x")
    axes[1].set_title("angles_gamma_g_x")
    axes[2].set_title("angles_g_gamma_x")
    fig.suptitle(f"{args.exp_name}\ncheck_stoc_equivariance_gamma_hist: use_model={use_model}")

    save_fig(args, fig)


def plot_system_on_axes(x, axis, label, same_plot=False):
    # [n, 2, 2]
    pass


def check_stoc_equivariance(args, iter_dataloader, generative_model, n=100, use_gamma=True, plot_gamma=False):

    x, t = get_x_t(args, iter_dataloader, generative_model, add_noise=True)

    # [1, 2, 2]
    g = orthogonal_haar(
        dim=2, target_tensor=torch.empty(1, device=args.device)
    )

    # [1, 2, 3] 
    xh = convert_x_to_xh(x)
    g_xh = convert_x_to_xh(torch.bmm(x, g.transpose(1, 2)))

    # [n, 2, 2]
    if use_gamma:
        fx, gamma_x = sample_multiple_outputs(args, xh, t, generative_model, n=n, output_type="model")
        fg_x, gamma_g_x = sample_multiple_outputs(args, g_xh, t, generative_model, n=n, output_type="model")
    else:
        fx = sample_multiple_outputs(args, xh, t, generative_model, n=n, output_type="k")
        fg_x = sample_multiple_outputs(args, g_xh, t, generative_model, n=n, output_type="k")

    if plot_gamma:
        assert use_gamma is True
        g_gamma_x = torch.bmm(g.repeat_interleave(n, dim=0), gamma_x)
    else:
        g_fx = torch.bmm(fx, g.repeat_interleave(n, dim=0).transpose(1, 2))

    # plots
    fig, axes = plt.subplots(1, 2, figsize=(15, 8))

    xh, g_xh = convert_cpu_detach(xh, g_xh)

    # x, g_x
    axes[0].scatter(xh[:, :, 0], xh[:, :, 1], label="x")
    axes[0].scatter(g_xh[:, :, 0], g_xh[:, :, 1], label="g_x")

    if plot_gamma:
        gamma_x, gamma_g_x, g_gamma_x = convert_cpu_detach(gamma_x, gamma_g_x, g_gamma_x)

        axes[0].scatter(gamma_x[:, 0, 0], gamma_x[:, 1, 0], alpha=0.2, label="gamma_x")
        axes[0].scatter(gamma_g_x[:, 0, 0], gamma_g_x[:, 1, 0], alpha=0.2, label="gamma_g_x")
        axes[0].scatter(g_gamma_x[:, 0, 0], g_gamma_x[:, 1, 0], alpha=0.2, label="g_gamma_x")

    else:
        fx, fg_x, g_fx = convert_cpu_detach(fx, fg_x, g_fx)

        # f(x)
        axes[0].scatter(fx[:, 0, 0], fx[:, 0, 1], alpha=0.2, label="f(x)_1")
        axes[0].scatter(fx[:, 1, 0], fx[:, 1, 1], alpha=0.2, label="f(x)_2")

        # f(g_x)
        axes[0].scatter(fg_x[:, 0, 0], fg_x[:, 0, 1], alpha=0.2, label="f(g_x)_1")
        axes[0].scatter(fg_x[:, 1, 0], fg_x[:, 1, 1], alpha=0.2, label="f(g_x)_2")

        # g_f(x)
        axes[0].scatter(g_fx[:, 0, 0], g_fx[:, 0, 1], alpha=0.2, label="g_f(x)_1")
        axes[0].scatter(g_fx[:, 1, 0], g_fx[:, 1, 1], alpha=0.2, label="g_f(x)_2")

        # first component of f(g_x) and g_f(x)
        axes[1].scatter(fg_x[:, 0, 0], fg_x[:, 0, 1], alpha=0.2, label="f(g_x)_1")
        axes[1].scatter(g_fx[:, 0, 0], g_fx[:, 0, 1], alpha=0.2, label="g_f(x)_1")

    axes[0].legend()
    axes[1].legend()

    fig.suptitle(f"{args.exp_name}\ncheck_stoc_equivariance: t={t[0].item()}, n={n}, use_gamma={use_gamma}, plot_gamma={plot_gamma}")
    save_fig(args, fig)


def sample_timestep(args, n_times, generative_model, return_int=False):
    t_int = torch.randint(
        0, generative_model.T + 1, 
        size=(n_times,), 
        device=args.device
    ).float()

    if return_int:
        return t_int
    else:
        return t_int / generative_model.T  # [n_times]


def add_noise_x(args, xh, generative_model, t_int=None, fix_noise=False, return_t=False):
    # xh: [bs, 2, 2] or [bs, 2, 3]
    # t_int: [bs]

    bs = len(xh)
    node_mask = torch.ones(bs, 2, 1, device=args.device)

    if t_int is None:
        t = sample_timestep(args, bs, generative_model)
    else:
        t = t_int / generative_model.T
    t = t.unsqueeze(-1)  # [bs, 1]

    # [bs, 1, 1]
    gamma_t = generative_model.inflate_batch_array(generative_model.gamma(t), xh)
    alpha_t = generative_model.alpha(gamma_t, xh)
    sigma_t = generative_model.sigma(gamma_t, xh)

    n_samples = 1 if fix_noise else bs

    # samples eps_x com-free and eps_h
    eps = generative_model.sample_combined_position_feature_noise(
        n_samples=n_samples, n_nodes=2, node_mask=node_mask, com_free=args.com_free
    )[:, :, :2]

    noised_x = alpha_t * xh[:, :, :2] + sigma_t * eps

    assert_mean_zero_with_mask(noised_x, node_mask)

    if return_t:
        return noised_x, t
    else:
        return noised_x


def sample_xh_batch_dataloader(args, iter_dataloader):

    batch = next(iter_dataloader)

    x = batch["positions"].to(args.device) 
    h = torch.zeros(len(x), 2, 1, device=args.device)

    return torch.cat([x, h], dim=-1)  # [bs, 2, 3]
    

def get_x_t(args, iter_dataloader, generative_model, add_noise=True):
    xh = sample_xh_batch_dataloader(args, iter_dataloader)[[0]]  # [1, 2, 3]
    t = sample_timestep(args, 1, generative_model)
    if add_noise:
        xh = add_noise_x(args, xh, generative_model)  # [1, 2, 2]
    return xh[:, :, :2], t


def visualise_p_t(args, iter_train_dataloader, generative_model):

    xh = sample_xh_batch_dataloader(args, iter_train_dataloader)
    t_int = sample_timestep(args, 1, generative_model, return_int=True).repeat_interleave(len(xh))
    noised_x = add_noise_x(args, xh, generative_model, t_int=t_int, fix_noise=False)
    plot_positions(args, noised_x.cpu().detach().numpy(), project=False, 
                   plot_single=False, title=f"p_t at t_int={t_int[0].item()}")


def convert_x_to_xh(x):
    h = torch.zeros(len(x), 2, 1, device=x.device)
    return torch.cat([x, h], dim=-1)


def main():
    
    parser = argparse.ArgumentParser(description='eval_toy_experiment')
    parser.add_argument('--model_path', type=str, default="")
    parser.add_argument('--plots_path', type=str, default="/data/ziz/not-backed-up/lezhang/projects/symdiff/plots")
    parser.add_argument('--n_samples', type=int, default=100)

    parser.add_argument('--visualise_data', action="store_true")

    parser.add_argument('--visualise_samples', action="store_true")
    
    parser.add_argument('--visualise_gamma', action="store_true")

    parser.add_argument('--visualise_gamma_hist', action="store_true")

    parser.add_argument('--visualise_stoc_eq', action="store_true")

    parser.add_argument('--model_equivariance_metric', action="store_true")
    parser.add_argument('--n_importance_samples', type=int, default=1)

    parser.add_argument('--return_iwae_nll', action="store_true")
    
    eval_args = parser.parse_args()

    with open(os.path.join(eval_args.model_path, 'args.pickle'), 'rb') as f:
        args = pickle.load(f)

    args.resample_toy_data = False  # to ensure we use the same dataset as training

    # create folder for saving plots
    fig_folder_datatime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fig_folder_name = "_".join([fig_folder_datatime, args.exp_name, args.dataset])
    args.fig_folder_path = os.path.join(eval_args.plots_path, fig_folder_name)
    try:
        os.makedirs(args.fig_folder_path)
    except OSError:
        pass

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")
    args.device = device
    dtype = torch.float32

    # load saved model
    generative_model, _, _ = get_model(args, device, None, None)
    generative_model.to(device)
    print(f"\nLoaded model from: {eval_args.model_path}\n")

    fn = 'generative_model_ema.npy' if args.ema_decay > 0 else 'generative_model.npy'
    flow_state_dict = torch.load(os.path.join(eval_args.model_path, fn), map_location=device)
    generative_model.load_state_dict(flow_state_dict)
    generative_model.eval()

    # load datasets
    #print("CHANGING BATCHSIZE")
    #args.batch_size = 16
    dataloaders = retrieve_dataloaders(args)
    iter_train_dataloader = iter(dataloaders["train"])
    iter_val_dataloader = iter(dataloaders["valid"])

    print(f"valid dataset has len={len(dataloaders['valid'].dataset.data["positions"])}")


    if eval_args.visualise_data:
        # visualise training dataset
        print("\nplotting training dataset")
        train_positions = dataloaders["train"].dataset.data["positions"]
        plot_positions(args, train_positions, project=False, 
                       plot_single=False, title="train_dataset")
        plot_positions(args, train_positions, project=True, 
                       plot_single=False, title="train_dataset radius histogram")

        # visualise p_t
        print("\nplotting p_t")
        visualise_p_t(args, iter_train_dataloader, generative_model)

    # visualise samples
    if eval_args.visualise_samples:
        print("\nplotting samples from model")
        samples = sample(eval_args, generative_model, fix_noise=False, return_np=True)
        plot_positions(args, samples, project=False, plot_single=False)
        plot_radius(args, samples)
        plot_positions_1d(args, samples)

    # visualise chain
    #chain = sample_chain(eval_args, generative_model)
    #plot_positions(chain, project=False, plot_single=False)
    #plot_positions(chain, project=True, plot_single=False)

    # plot samples from gamma
    if eval_args.visualise_gamma:
        print("\nplotting gamma samples")
        xh = sample_xh_batch_dataloader(args, iter_train_dataloader)[[0]]
        x, t = add_noise_x(args, xh, generative_model, return_t=True)
        print(f"x_1_norm={torch.norm(x[:, 0])}, x_2_norm={torch.norm(x[:, 1])}")
        xh = convert_x_to_xh(x)
        plot_outputs(args, xh, t, generative_model, n=100, plot_gamma=True) 

    # plot gamma distribution
    if eval_args.visualise_gamma_hist:
        print("\nplotting histogram of gamma")
        plot_gamma_hist(args, iter_val_dataloader, generative_model, n=1000)
        check_stoc_equivariance_gamma_hist(args, iter_val_dataloader, generative_model, n=1000, use_model=False)

    # check stochastic equivariance
    if eval_args.visualise_stoc_eq:
        print("\nplotting stochastic equivariance\n")
        check_stoc_equivariance(args, iter_val_dataloader, generative_model, n=2000)  # check equivariance of symdiff
        check_stoc_equivariance(args, iter_val_dataloader, generative_model, n=10, use_gamma=False)  # check equivariance of k
        check_stoc_equivariance(args, iter_val_dataloader, generative_model, n=5000, use_gamma=True, plot_gamma=True)  # check equivariance of gamma

    if eval_args.model_equivariance_metric:
        print("\ncomputing model equivariance metric\n")
        args.use_equivariance_metric = True
        args.return_iwae_nll = False
        args.n_importance_samples = eval_args.n_importance_samples
        nll, model_metric, backbone_metric, _ = test(
            args, dataloaders["valid"], 0, generative_model, 
            device, dtype, None, None, partition='Valid')
        print(f"\nmodel metric: {model_metric}, backbone metric: {backbone_metric}")

    if eval_args.return_iwae_nll:
        print(f"\ncomputing iwae nll with n={eval_args.n_importance_samples}\n")
        args.use_equivariance_metric = False
        args.return_iwae_nll = True
        args.n_importance_samples = eval_args.n_importance_samples
        nll, _, _, iwae_nll = test(
            args, dataloaders["valid"], 0, generative_model, device, 
            dtype, None, None, partition="Valid")
        print(f"\nnll: {nll}, iwae_nll: {iwae_nll}")


if __name__ == "__main__":
    main()