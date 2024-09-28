"""
This is a script to extract the EMA and normal trained DiT model from a Dit_Gaussian_Dynamics model.


The reason this is needed is because the normal saving of the model saves everything including the embeddings models
for the main DiT model.
"""


try:
    from rdkit import Chem
    print("RDKit found and imported")
except ModuleNotFoundError:
    print("RDKit not found, please install it")

import copy
import utils
import argparse
import wandb
import torch
import time
import pickle
import os

from tqdm import tqdm
from os.path import join

from qm9 import dataset
from qm9.models import get_optim, get_scheduler, get_model
from qm9.utils import prepare_context, compute_mean_mad
from equivariant_diffusion import en_diffusion
from equivariant_diffusion.utils import assert_correctly_masked
from equivariant_diffusion import utils as flow_utils
from equivariant_diffusion.utils import assert_mean_zero_with_mask, remove_mean_with_mask,\
    assert_correctly_masked, sample_center_gravity_zero_gaussian_with_mask

from train_test import train_epoch, test, analyze_and_save
from configs.datasets_config import get_dataset_info


print(f"os.getcwd(): {os.getcwd()}")

############################################################################################################
# ARGUMENTS
############################################################################################################

parser = argparse.ArgumentParser(description='E3Diffusion')
parser.add_argument('--exp_name', type=str, default='debug_10')
parser.add_argument('--model', type=str, default='egnn_dynamics',
                    help='our_dynamics | schnet | simple_dynamics | '
                         'kernel_dynamics | egnn_dynamics |gnn_dynamics')
parser.add_argument('--probabilistic_model', type=str, default='diffusion',
                    help='diffusion')

# Training complexity is O(1) (unaffected), but sampling complexity is O(steps).
parser.add_argument('--diffusion_steps', type=int, default=500)
parser.add_argument('--diffusion_noise_schedule', type=str, default='polynomial_2',
                    help='learned, cosine')
parser.add_argument('--diffusion_noise_precision', type=float, default=1e-5,
                    )
parser.add_argument('--diffusion_loss_type', type=str, default='l2',
                    help='vlb, l2')

# -------- optimiser args -------- #

parser.add_argument("--use_separate_optimisers", action="store_true", help="Whether to use two separate optimizers for the K and Gamma",
                    default=False)

# Args for the optimiser for K
parser.add_argument('--lr_K', type=float, default=2e-4)
parser.add_argument('--use_amsgrad_K', action="store_true")  # default from EDM
parser.add_argument('--weight_decay_K', type=float, default=1e-12)  # default from EDM

# Args for the optimiser for gamma
parser.add_argument('--lr_gamma', type=float, default=2e-4)
parser.add_argument('--use_amsgrad_gamma', action="store_true")  # default from EDM
parser.add_argument('--weight_decay_gamma', type=float, default=1e-12)  # default from EDM

# Other args
parser.add_argument('--n_epochs', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--use_amsgrad', action="store_true")  # default from EDM
parser.add_argument('--weight_decay', type=float, default=1e-12)  # default from EDM

parser.add_argument('--scheduler', type=str, default=None)  # default from EDM
parser.add_argument('--num_warmup_steps', type=int, default=30000)  # default from EDM
parser.add_argument('--num_training_steps', type=int, default=350000)  # default from EDM

parser.add_argument('--clipping_type', type=str, default="queue")
parser.add_argument('--max_grad_norm', type=float, default=1.5)

parser.add_argument('--brute_force', type=eval, default=False,
                    help='True | False')
parser.add_argument('--actnorm', type=eval, default=True,
                    help='True | False')
parser.add_argument('--break_train_epoch', type=eval, default=False,
                    help='True | False')
parser.add_argument('--dp', type=eval, default=True,
                    help='True | False')
parser.add_argument('--condition_time', type=eval, default=True,
                    help='True | False')
parser.add_argument('--clip_grad', type=eval, default=True,
                    help='True | False')
parser.add_argument('--trace', type=str, default='hutch',
                    help='hutch | exact')

# EGNN args -->
parser.add_argument('--n_layers', type=int, default=6,
                    help='number of layers')
parser.add_argument('--inv_sublayers', type=int, default=1,
                    help='number of layers')
parser.add_argument('--nf', type=int, default=128,
                    help='number of layers')
parser.add_argument('--tanh', type=eval, default=True,
                    help='use tanh in the coord_mlp')
parser.add_argument('--attention', type=eval, default=True,
                    help='use attention in the EGNN')
parser.add_argument('--norm_constant', type=float, default=1,
                    help='diff/(|diff| + norm_constant)')
parser.add_argument('--sin_embedding', type=eval, default=False,
                    help='whether using or not the sin embedding')
# <-- EGNN args
parser.add_argument('--ode_regularization', type=float, default=1e-3)
parser.add_argument('--dataset', type=str, default='qm9',
                    help='qm9 | qm9_second_half (train only on the last 50K samples of the training dataset)')
parser.add_argument('--datadir', type=str, default='qm9/temp',
                    help='qm9 directory')
parser.add_argument('--force_download', type=bool, default=False, help='Whether to redownload qm9')  # LEO
parser.add_argument('--filter_n_atoms', type=int, default=None,
                    help='When set to an integer value, QM9 will only contain molecules of that amount of atoms')
parser.add_argument('--dequantization', type=str, default='argmax_variational',
                    help='uniform | variational | argmax_variational | deterministic')
parser.add_argument('--n_report_steps', type=int, default=1)
parser.add_argument('--wandb_usr', type=str)
parser.add_argument('--no_wandb', action='store_true', help='Disable wandb')
parser.add_argument('--online', type=bool, default=True, help='True = wandb online -- False = wandb offline')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--save_model', type=eval, default=True,
                    help='save model')
parser.add_argument('--generate_epochs', type=int, default=1,
                    help='save model')
parser.add_argument('--num_workers', type=int, default=0, help='Number of worker for the dataloader')
parser.add_argument('--test_epochs', type=int, default=10)
parser.add_argument('--data_augmentation', type=eval, default=False, help='use attention in the EGNN')
parser.add_argument("--conditioning", nargs='+', default=[],
                    help='arguments : homo | lumo | alpha | gap | mu | Cv' )
parser.add_argument('--resume', type=str, default=None,
                    help='')
parser.add_argument('--start_epoch', type=int, default=0,
                    help='')
parser.add_argument('--ema_decay', type=float, default=0.999,
                    help='Amount of EMA decay, 0 means off. A reasonable value'
                         ' is 0.999.')
parser.add_argument('--augment_noise', type=float, default=0)
parser.add_argument('--n_stability_samples', type=int, default=500,
                    help='Number of samples to compute the stability')
parser.add_argument('--normalize_factors', type=eval, default=[1, 4, 1],
                    help='normalize factors for [x, categorical, integer]')
parser.add_argument('--remove_h', action='store_true')
parser.add_argument('--include_charges', type=eval, default=True,
                    help='include atom charge or not')
parser.add_argument('--visualize_every_batch', type=int, default=1e8,
                    help="Can be used to visualize multiple times per epoch")
parser.add_argument('--normalization_factor', type=float, default=1,
                    help="Normalize the sum aggregation of EGNN")
parser.add_argument('--aggregation_method', type=str, default='sum',
                    help='"sum" or "mean"')

# Use separate ema
parser.add_argument('--use_separate_ema', type=eval, default=False,
                    help='Use separate ema for the gamma and k')                    
parser.add_argument('--ema_decay_K', type=float, default=0.999,
                    help='Amount of EMA decay, 0 means off. A reasonable value'
                         ' is 0.999.')
parser.add_argument('--ema_decay_gamma', type=float, default=0.999,
                    help='Amount of EMA decay, 0 means off. A reasonable value'
                         ' is 0.999.')

# -------- sym_diff args -------- #

parser.add_argument("--sigma_data", type=float, default=1.5, help="for VE scaling of inputs")
parser.add_argument("--com_free", action="store_true", help="whether to use the CoM subspace")
parser.add_argument("--rho", type=float, default=3, help="VE schedule")
parser.add_argument("--sigma_min", type=float, default=1e-5, help="VE schedule")
parser.add_argument("--sigma_max", type=float, default=10, help="VE schedule")

parser.add_argument("--print_grad_norms", action="store_true", help="whether to show the gamma and k grad norms")
parser.add_argument("--print_parameter_count", action="store_true", help="whether to show the gamma and k param count")

# -------- sym_diff perceiver args -------- #

parser.add_argument("--context_hidden_size", type=int, default=512, help="preprocessor config for perceiver")

parser.add_argument("--gamma_num_latents", type=int, default=64, help="gamma config for perceiver")
parser.add_argument("--gamma_d_latents", type=int, default=128, help="gamma config for perceiver")
parser.add_argument("--gamma_n_pad", type=int, default=61, help="gamma config for perceiver")
parser.add_argument("--gamma_num_blocks", type=int, default=1, help="gamma config for perceiver")
parser.add_argument("--gamma_num_self_attends_per_block", type=int, default=3, help="gamma config for perceiver")
parser.add_argument("--gamma_num_self_attention_heads", type=int, default=4, help="gamma config for perceiver")
parser.add_argument("--gamma_num_cross_attention_heads", type=int, default=4, help="gamma config for perceiver")
parser.add_argument("--gamma_attention_probs_dropout_prob", type=float, default=0.1, help="gamma config for perceiver")
parser.add_argument("--gamma_pos_num_channels", type=int, default=64, help="gamma config for perceiver")
parser.add_argument("--gamma_num_heads", type=int, default=4, help="gamma config for perceiver")

parser.add_argument("--k_num_latents", type=int, default=128, help="k config for perceiver")
parser.add_argument("--k_d_latents", type=int, default=256, help="k config for perceiver")
parser.add_argument("--k_n_pad", type=int, default=55, help="k config for perceiver")
parser.add_argument("--k_num_blocks", type=int, default=1, help="k config for perceiver")
parser.add_argument("--k_num_self_attends_per_block", type=int, default=10, help="k config for perceiver")
parser.add_argument("--k_num_self_attention_heads", type=int, default=4, help="k config for perceiver")
parser.add_argument("--k_num_cross_attention_heads", type=int, default=4, help="k config for perceiver")
parser.add_argument("--k_attention_probs_dropout_prob", type=float, default=0.1, help="k config for perceiver")
parser.add_argument("--k_enc_mlp_factor", type=int, default=2, help="k config for perceiver")

parser.add_argument("--k_pos_num_channels", type=int, default=64, help="k config for perceiver")
parser.add_argument("--k_num_heads", type=int, default=4, help="k config for perceiver")
parser.add_argument("--k_decoder_self_attention", action="store_true", help="k config for perceiver")
parser.add_argument("--k_num_self_heads", type=int, default=4, help="k config for perceiver")
parser.add_argument("--k_query_residual", action="store_true", help="k config for perceiver")

parser.add_argument("--decoder_hidden_size", type=int, default=256, help="k config for perceiver")

# -------- sym_diff transformer args -------- #

parser.add_argument("--gamma_num_enc_layers", type=int, default=2, help="gamma config for transformer")
parser.add_argument("--gamma_num_dec_layers", type=int, default=2, help="gamma config for transformer")
parser.add_argument("--gamma_d_model", type=int, default=128, help="gamma config for transformer")
parser.add_argument("--gamma_nhead", type=int, default=4, help="gamma config for transformer")
parser.add_argument("--gamma_dim_feedforward", type=int, default=256, help="gamma config for transformer")
parser.add_argument("--gamma_dropout", type=float, default=0.1, help="gamma config for transformer")

parser.add_argument("--k_num_layers", type=int, default=6, help="k config for transformer")
parser.add_argument("--k_d_model", type=int, default=256, help="k config for transformer")
parser.add_argument("--k_nhead", type=int, default=8, help="k config for transformer")
parser.add_argument("--k_dim_feedforward", type=int, default=512, help="k config for transformer")
parser.add_argument("--k_dropout", type=float, default=0.1, help="k config for transformer")

# -------- sym_diff perceiver fourier args -------- #

parser.add_argument("--sigma", type=float, default=100, help="config for perceiver fourier")
parser.add_argument("--m", type=int, default=20, help="config for perceiver fourier")

# -------- perceiver_gaussian args -------- #

parser.add_argument("--pos_emb_size", type=int, default=256, help="config for perceiver fourier")
parser.add_argument("--k_mlp_factor", type=int, default=2, help="config for perceiver fourier")


# -------- transformer args -------- #

parser.add_argument("--trans_num_layers", type=int, default=6, help="config for transformer")
parser.add_argument("--trans_d_model", type=int, default=256, help="config for transformer")
parser.add_argument("--trans_nhead", type=int, default=8, help="config for transformer")
parser.add_argument("--trans_dim_feedforward", type=int, default=512, help="config for transformer")
parser.add_argument("--trans_dropout", type=float, default=0., help="config for transformer")

# -------- DiT args -------- #

parser.add_argument("--out_channels", type=int, default=9, help="config for DiT")
parser.add_argument("--x_scale", type=float, default=25.0, help="config for DiT")
parser.add_argument("--hidden_size", type=int, default=256, help="config for DiT")
parser.add_argument("--depth", type=int, default=6, help="config for DiT")
parser.add_argument("--num_heads", type=int, default=4, help="config for DiT")
parser.add_argument("--mlp_ratio", type=float, default=2.0, help="config for DiT")
parser.add_argument("--subtract_x_0", action="store_true", help="config for DiT")

parser.add_argument("--x_emb", type=str, default="fourier", help="config for DiT")

# -------- DiT_GNN and DiT_DiT args -------- #

parser.add_argument("--enc_out_channels", type=int, default=1, help="config for DiT_GNN")  # not used
parser.add_argument("--enc_x_scale", type=float, default=25.0, help="config for DiT_GNN")
parser.add_argument("--enc_hidden_size", type=int, default=64, help="config for DiT_GNN")
parser.add_argument("--enc_depth", type=int, default=4, help="config for DiT_GNN")
parser.add_argument("--enc_num_heads", type=int, default=4, help="config for DiT_GNN")
parser.add_argument("--enc_mlp_ratio", type=float, default=4, help="config for DiT_GNN")
parser.add_argument("--dec_hidden_features", type=int, default=32, help="config for DiT_GNN")

parser.add_argument("--enc_x_emb", type=str, default="linear", help="config for DiT_GNN and DiT_DiT")
parser.add_argument("--enc_concat_h", action="store_true", help="config for DiT_GNN")
parser.add_argument("--noise_dims", type=int, default=16, help="config for DiT_GNN")
parser.add_argument("--noise_std", type=float, default=1.0, help="config for DiT_GNN")

# -------- DiTGaussian_GNN args -------- #

parser.add_argument("--pos_size", type=int, default=128, help="config for DiTGaussian_GNN")

# -------- GNN_DiT args -------- #

parser.add_argument("--gamma_gnn_layers", type=int, default=4, help="config for GNN_GNN")
parser.add_argument("--gamma_gnn_hidden_size", type=int, default=64, help="config for GNN_GNN")
parser.add_argument("--gamma_gnn_out_size", type=int, default=64, help="config for GNN_GNN")
parser.add_argument("--gamma_dec_hidden_size", type=int, default=32, help="config for GNN_GNN")

# -------- DiTEmb args -------- #

parser.add_argument("--xh_hidden_size", type=int, default=128, help="config for DiTEmb")
parser.add_argument("--mlp_dropout", type=float, default=0.0, help="config for DiTEmb")


# Seprate arguments for dropout        
parser.add_argument("--use_separate_dropout", action="store_true", help="Whether to use separate dropouts for gamma enc, gamma dec, and k")
parser.add_argument("--dropout_gamma_enc", type=float, default=0.0, help="Dropout for gamma encoder")
parser.add_argument("--dropout_gamma_dec", type=float, default=0.0, help="Dropout for gamma decoder")
parser.add_argument("--dropout_k", type=float, default=0.0, help="Dropout for k")


# -------- DiTGaussian args -------- #

parser.add_argument("--K", type=int, default=128, help="config for DiTGaussian")
parser.add_argument("--mlp_type", type=str, default="mlp", help="config for DiTGaussian")

# Positional embeddings
parser.add_argument("--use_separate_gauss_embs", action="store_true", help="Whether to use separate Gaussian embeddings for k and gamma")
parser.add_argument("--gamma_K", type=float, default=0, help="K for gamma positional embeddings")
parser.add_argument("--k_K", type=float, default=0, help="K for k positional embeddings")
parser.add_argument("--pos_emb_gamma_projection_dim", type=float, default=0, help="Dimension of the projection for gamma positional embeddings")




# -------- sym_diff time args -------- #

parser.add_argument("--enc_gnn_layers", type=int, default=2, help="config for DiTMessage")
parser.add_argument("--enc_gnn_hidden_size", type=int, default=256, help="config for DiTMessage")

# -------- sym_diff time args -------- #

parser.add_argument("--num_bands", type=int, default=32, help="fourier time embedding config")
parser.add_argument("--max_resolution", type=float, default=100, help="fourier time embedding config")
parser.add_argument("--concat_t", action="store_true", help="fourier time embedding config")
parser.add_argument("--t_fourier", action="store_true", help="time config for transformer")


### Model parts to freeze for DIT DIT model  ####
parser.add_argument("--freeze_model_parts", action="store_true", help="Whether to freeze the model parts")
parser.add_argument("--model_part_to_freeze", type=str, default="", help="Which part of the model to freeze")
parser.add_argument("--path_to_load_backbone", type=str, default="", help="Path to load the backbone model from, if loading only the backbone")
parser.add_argument("--type_backbone_to_load", type=str, default="", help="Whether to load the EMA backbone or the normal backbone: EMA or normal")


# Data aug at sampling
parser.add_argument("--data_aug_at_sampling", action="store_true", help="Whether to augment data at sampling time")

# Arguments for equivariance metrics
parser.add_argument("--use_equivariance_metric", action="store_true", help="whether to log the equivariance metrics")
parser.add_argument("--n_importance_samples", type=int, default=32, help="whether to log the equivariance metrics")
parser.add_argument('--n_dims', type=int, default=3)

############################################################################################################
# NEW ARGS
############################################################################################################

parser.add_argument('--model_loc', type=str, default="Location of DiT Gaussian Dynamics model")

# Arguments for stochasticiy
parser.add_argument("--gamma_samples_stochasticity", type=int, default=5000, help="Number of samples to check the stochasticity of gamma")

############################################################################################################



############################################################################################################
# GET NORMAL MODEL AND DATASET
############################################################################################################

# Getting the dataset
args = parser.parse_args()
print(args)
dataset_info = get_dataset_info(args.dataset, args.remove_h)  # get configs for qm9 etc.
atom_encoder = dataset_info['atom_encoder']
atom_decoder = dataset_info['atom_decoder']


# Getting args
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
dtype = torch.float32

# Retrieve QM9 dataloaders
dataloaders, charge_scale = dataset.retrieve_dataloaders(args)
data_dummy = next(iter(dataloaders['train']))

# Getting conditioning info
if len(args.conditioning) > 0:
    print(f'Conditioning on {args.conditioning}')
    property_norms = compute_mean_mad(dataloaders, args.conditioning, args.dataset)  # compute mean, mad of each prop
    context_dummy = prepare_context(args.conditioning, data_dummy, property_norms)  # combines conditioning info into tensor
    context_node_nf = context_dummy.size(2)  # this is combined dim of all conditioning props
else:
    context_node_nf = 0
    property_norms = None

args.context_node_nf = context_node_nf


# Create the model (DiT Gaussian Dynamics)
model, nodes_dist, prop_dist = get_model(args, device, dataset_info, dataloaders['train'])
if prop_dist is not None:  # when conditioning
    prop_dist.set_normalizer(property_norms)
model = model.to(device)

# Load the full model from model_loc
flow_state_dict = torch.load(args.model_loc)
model.load_state_dict(flow_state_dict)


############################################################################################################
# GET EMA MODEL
############################################################################################################

# Get the EMA model and save it

# Create the model (DiT Gaussian Dynamics) again
model_ema, _, _ = get_model(args, device, dataset_info, dataloaders['train'])
if prop_dist is not None:  # when conditioning
    prop_dist.set_normalizer(property_norms)
model_ema = model_ema.to(device)


# Load the full model from model_loc - replace generative_model.npy with  generative_model_ema
ema_state_dict = torch.load(args.model_loc.replace("generative_model.npy", "generative_model_ema.npy"))
model_ema.load_state_dict(ema_state_dict)

############################################################################################################
# GET STOCHAISTY OF GAMMA
############################################################################################################

# To get stochasticity of gamma:

# 1. Load the dataloader of our QM9 dataset.

# 2. Get one datapoint from the QM9 dataset.

# 3. Put this datapoint into the tensor format that our model expect.

# 4. Create gamma_samples_stochasticity number of samples of the same molecule

# 5. Create a node mask of size (gamma_samples_stochasticity, 1, 1) where all elements are 1s

# 6. Pass this tensor through the _forward model of our DDG model, where for the DDG model we have an argument to output just the gammas
#     a. This outputs a tensor of size (gamma_samples_stochasticity, 3, 3) which is the gamma tensor

# 7. Subset the outputted tensor to get a tensor of shape (gamma_samples_stochasticity, 3, 1)

# 8. Save this as a numpy array called "stochastic_gamma_samples" in the experiment name folder


# Getting one datapoint from the QM9 test dataloader

model.eval()
model_ema.eval()
test_loader = dataloaders['test']
n_iterations = len(test_loader)
dtype = torch.float32
for i, data in tqdm(test_loader):
    x = data["position"].to(device, dtype) 

    node_mask = data['atom_mask'].to(device, dtype).unsqueeze(2)  # [bs, n_nodes, 1]
    edge_mask = data['edge_mask'].to(device, dtype)  # [bs*n_nodes^2, 1]
    one_hot = data['one_hot'].to(device, dtype)  # [bs, n_nodes, num_classes - i.e. 5 for qm9]
    charges = (data['charges'] if args.include_charges else torch.zeros(0)).to(device, dtype)  # [bs, n_nodes, 1]

    x = remove_mean_with_mask(x, node_mask)

    # Print the shape of all of the above objecvts in this for loop
    print(f"x: {x.shape}")
    print(f"node_mask: {node_mask.shape}")
    print(f"edge_mask: {edge_mask.shape}")
    print(f"one_hot: {one_hot.shape}")
    print(f"charges: {charges.shape}")


    # Print batch size
    bs = x.size(0)

