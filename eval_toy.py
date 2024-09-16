import argparse
import os
import pickle
import datetime

import torch
import numpy as np
import matplotlib.pyplot as plt


from qm9.models import get_model
from equivariant_diffusion.utils import assert_mean_zero_with_mask, remove_mean_with_mask, assert_correctly_masked

from sym_nn.distributions import retrieve_dataloaders
from sym_nn.utils import orthogonal_haar


def save_fig(args, fig):
    dt = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # for saving plots etc.
    fig_name = "_".join([args.exp_name, args.dataset, dt])
    fig_path = os.path.join(
        args.fig_path, fig_name
    )
    fig.savefig(fig_path)
    plt.close()


def plot_positions(args, positions, project=False, plot_single=False, title=None):
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

    if title is not None:
        axes.set_title(title)

    save_fig(args, fig)


def plot_radius(args, positions):
    radius_0 = np.linalg.norm(positions[:, 0], axis=-1)  # [n]
    radius_1 = np.linalg.norm(positions[:, 1], axis=-1)  # [n]
    radius_diff = np.sqrt((radius_0 - radius_1)**2)
    fig, axes = plt.subplots(1, 1)
    _ = axes.hist(radius_diff, bins=100)
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


def sample_outputs(args, xh, t, generative_model, n=5, plot=True):
    # xh: [1, 2, 3]
    # t: [1]

    x_0 = xh[:, :, :2]
    xh = xh.repeat_interleave(n, dim=0)  # [n, 2, 3]
    t = t.repeat_interleave(n)
    node_mask = torch.ones(n, 2, 1, device=args.device)

    xh, gamma = generative_model.dynamics._forward(
        t, xh, node_mask, None, None, return_gamma=True
    )
    x = xh[:, :, :2]  # [n, 2, 2]
    angles = gamma[:, :, 0]  # from action on (1, 0) on [n, 2, 2]

    if plot:
        print(f"x: {x_0}")
        print(f"t: {t}")

        x = x.cpu().detach().numpy()
        angles = angles.cpu().detach().numpy()

        fig, axes = plt.subplots(1, 1)

        axes.scatter(x_0.cpu()[:, 0, 0], x_0.cpu()[:, 0, 1], label="x0_1", c="black")
        axes.scatter(x_0.cpu()[:, 1, 0], x_0.cpu()[:, 1, 1], label="x0_2", c="yellow")

        axes.scatter(x[:, 0, 0], x[:, 0, 1], alpha=0.5, label="x_1")
        axes.scatter(x[:, 1, 0], x[:, 1, 1], alpha=0.5, label="x_2")

        axes.scatter(angles[:, 0], angles[:, 1], alpha=0.5, label="gamma")

        fig.suptitle(f"t={t[0].item()}")
        axes.legend()
        axes.set_aspect('equal')
        save_fig(args, fig)

    return x, angles


def convert_cpu_detach(*args):
    np_args = (arg.cpu().detach() for arg in args)
    return np_args


def check_stoc_equivariance(args, iter_dataloader, generative_model, n=100):

    xh = sample_xh_batch_dataloader(args, iter_dataloader)[[0]]  # [1, 2, 3]
    t = sample_timestep(args, 1, generative_model)
    noised_x = add_noise_x(args, xh, generative_model)

    # [1, 2, 2]
    g = orthogonal_haar(
        dim=2, target_tensor=torch.empty(1, device=args.device)
    )

    # [1, 2, 3] 
    xh = convert_x_to_xh(noised_x)
    g_xh = convert_x_to_xh(torch.bmm(noised_x, g.transpose(1, 2)))

    # [n, 2, 2]
    fx, _ = sample_outputs(args, xh, t, generative_model, n=n, plot=False)
    fg_x, _ = sample_outputs(args, g_xh, t, generative_model, n=n, plot=False)

    g_fx = torch.bmm(fx, g.repeat_interleave(n, dim=0).transpose(1, 2))

    # plots
    fig, axes = plt.subplots(1, 2, figsize=(15, 8))

    xh, fx = convert_cpu_detach(xh, fx)
    g, g_xh, fg_x, g_fx = convert_cpu_detach(g, g_xh, fg_x, g_fx)

    # x, f(x)
    axes[0].scatter(xh[:, :, 0], xh[:, :, 1], label="x")
    axes[0].scatter(fx[:, 0, 0], fx[:, 0, 1], alpha=0.5, label="f(x)_1")
    axes[0].scatter(fx[:, 1, 0], fx[:, 1, 1], alpha=0.5, label="f(x)_2")

    # g_x, f(g_x)
    axes[0].scatter(g_xh[:, :, 0], g_xh[:, :, 1], label="g_x")
    axes[0].scatter(fg_x[:, 0, 0], fg_x[:, 0, 1], alpha=0.5, label="f(g_x)_1")
    axes[0].scatter(fg_x[:, 1, 0], fg_x[:, 1, 1], alpha=0.5, label="f(g_x)_2")

    # g_f(x)
    axes[0].scatter(g_fx[:, 0, 0], g_fx[:, 0, 1], alpha=0.5, label="g_f(x)_1")
    axes[0].scatter(g_fx[:, 1, 0], g_fx[:, 1, 1], alpha=0.5, label="g_f(x)_2")

    axes[0].legend()

    # first component of f(g_x) and g_f(x)
    axes[1].scatter(fg_x[:, 0, 0], fg_x[:, 0, 1], alpha=0.5, label="f(g_x)_1")
    axes[1].scatter(g_fx[:, 0, 0], g_fx[:, 0, 1], alpha=0.5, label="g_f(x)_1")
    
    axes[1].legend()

    fig.suptitle(f"t={t[0].item()}, n={n}")
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
    parser.add_argument('--n_samples', type=int, default=100)

    eval_args = parser.parse_args()

    with open(os.path.join(eval_args.model_path, 'args.pickle'), 'rb') as f:
        args = pickle.load(f)

    args.resample_toy_data = False  # to ensure we use the same dataset as training
    args.datatime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # for saving plots etc.
    args.fig_path = "/data/ziz/not-backed-up/lezhang/projects/symdiff/plots"

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")
    args.device = device
    dtype = torch.float32

    # load saved model and datasets
    generative_model, _, _ = get_model(args, device, None, None)
    generative_model.to(device)
    print(f"Loaded model from: {eval_args.model_path}")

    fn = 'generative_model_ema.npy' if args.ema_decay > 0 else 'generative_model.npy'
    flow_state_dict = torch.load(os.path.join(eval_args.model_path, fn), map_location=device)
    generative_model.load_state_dict(flow_state_dict)
    generative_model.eval()

    dataloaders = retrieve_dataloaders(args)
    iter_train_dataloader = iter(dataloaders["train"])
    iter_val_dataloader = iter(dataloaders["valid"])

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
    print("\nplotting samples from model")
    samples = sample(eval_args, generative_model, fix_noise=False, return_np=True)
    plot_positions(args, samples, project=False, plot_single=False, title="samples")
    plot_radius(samples)

    # visualise chain
    #chain = sample_chain(eval_args, generative_model)
    #plot_positions(chain, project=False, plot_single=False)
    #plot_positions(chain, project=True, plot_single=False)

    # plot samples from gamma
    print("\nplotting gamma samples")
    xh = sample_xh_batch_dataloader(args, iter_train_dataloader)[[0]]
    x, t = add_noise_x(args, xh, generative_model, return_t=True)
    print(f"x_1_norm={torch.norm(x[:, 0])}, x_2_norm={torch.norm(x[:, 1])}")
    xh = convert_x_to_xh(x)
    sample_outputs(args, xh, t, generative_model, n=10, plot=True) 

    # check stochastic equivariance
    print("\nplotting stochastic equivariance")
    check_stoc_equivariance(args, iter_val_dataloader, generative_model, n=100)

    # need to implement equivariance metrics, and k with gaussian noise

    # check if the gaussian noise is being used, 

if __name__ == "__main__":
    main()