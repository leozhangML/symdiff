# Rdkit import should be first, do not move it
try:
    from rdkit import Chem
except ModuleNotFoundError:
    pass
import copy
import utils
import argparse
import wandb
from configs.datasets_config import get_dataset_info
from os.path import join
from qm9 import dataset
from qm9.models import get_optim, get_scheduler, get_model
from equivariant_diffusion import en_diffusion
from equivariant_diffusion.utils import assert_correctly_masked
from equivariant_diffusion import utils as flow_utils
import torch
import time
import pickle
from qm9.utils import prepare_context, compute_mean_mad
from train_test import train_epoch, test, analyze_and_save

import os
print(f"os.getcwd(): {os.getcwd()}")

#wandb.login(key="78d61cd721affd9ffa2f5e217ed6f49de71eb842")

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

# -------- sym_diff args -------- #

parser.add_argument("--sigma_data", type=float, default=1.5, help="for VE scaling of inputs")
parser.add_argument("--com_free", action="store_true", help="whether to use the CoM subspace")
parser.add_argument("--rho", type=float, default=3, help="VE schedule")
parser.add_argument("--sigma_min", type=float, default=1e-5, help="VE schedule")
parser.add_argument("--sigma_max", type=float, default=10, help="VE schedule")

parser.add_argument("--print_grad_norms", action="store_true", help="whether to show the gamma and k grad norms")
parser.add_argument("--print_parameter_count", action="store_true", help="whether to show the gamma and k param count")

parser.add_argument("--use_equivariance_metric", action="store_true", help="whether to log the equivariance metrics")
parser.add_argument("--return_iwae_nll", action="store_true", help="whether to log the iwae estimates")
parser.add_argument("--n_importance_samples", type=int, default=10, help="whether to log the equivariance metrics")

parser.add_argument("--force_save_model", action="store_true", help="whether to force save the model at eval stage")
parser.add_argument("--save_model_at_init", action="store_true", help="whether to force save the model at random init")

# -------- DiTGaussian args -------- #

parser.add_argument("--xh_hidden_size", type=int, default=128, help="config for DiTGaussian")
parser.add_argument("--K", type=int, default=128, help="config for DiTGaussian")
parser.add_argument("--pos_embedder_test", type=int, default=4, help="config for DiTGaussian")


parser.add_argument("--hidden_size", type=int, default=256, help="config for DiT")
parser.add_argument("--depth", type=int, default=6, help="config for DiT")
parser.add_argument("--num_heads", type=int, default=4, help="config for DiT")
parser.add_argument("--mlp_ratio", type=float, default=2.0, help="config for DiT")
parser.add_argument("--mlp_dropout", type=float, default=0.0, help="config for DiTEmb")

# -------- DiT_DiTGaussian args -------- #

parser.add_argument("--enc_hidden_size", type=int, default=64, help="config for DiT_DiTGaussian")
parser.add_argument("--enc_depth", type=int, default=4, help="config for DiT_DiTGaussian")
parser.add_argument("--enc_num_heads", type=int, default=4, help="config for DiT_DiTGaussian")
parser.add_argument("--enc_mlp_ratio", type=float, default=4, help="config for DiT_DiTGaussian")

parser.add_argument("--dec_hidden_features", type=int, default=32, help="config for DiT_DiTGaussian")
parser.add_argument("--gamma_mlp_dropout", type=float, default=0.0, help="config for DiT_DiTGaussian")

parser.add_argument("--enc_concat_h", action="store_true", help="config for DiT_DiTGaussian")

parser.add_argument("--noise_dims", type=int, default=16, help="config for DiT_DiTGaussian")
parser.add_argument("--noise_std", type=float, default=1.0, help="config for DiT_DiTGaussian")

parser.add_argument("--mlp_type", type=str, default="mlp", help="config for DiT_DiTGaussian")

# -------- Deepsets DiTGaussian args -------- #

parser.add_argument("--pos_emb_gamma_size", type=int, default=32, help="config for Deepsets")
parser.add_argument("--t_hidden_size", type=int, default=32, help="config for Deepsets")

# -------- Scalars DiTGaussian args -------- #

parser.add_argument("--pos_emb_gamma_1_size", type=int, default=32, help="config for Deepsets")
parser.add_argument("--gamma_1_hidden_size", type=int, default=32, help="config for Deepsets")

# -------- ScalarsDiT_DiTGaussian args -------- #

parser.add_argument("--base_enc_hidden_size", type=int, default=64, help="config for ScalarsDiT_DiTGaussian")
parser.add_argument("--base_enc_depth", type=int, default=4, help="config for ScalarsDiT_DiTGaussian")
parser.add_argument("--base_enc_num_heads", type=int, default=4, help="config for ScalarsDiT_DiTGaussian")
parser.add_argument("--base_enc_mlp_ratio", type=float, default=4.0, help="config for ScalarsDiT_DiTGaussian")
parser.add_argument("--xh_gamma_hidden_size", type=int, default=128, help="config for DiTGaussian")


args = parser.parse_args()
args.molecule = True
args.n_dims = 3

dataset_info = get_dataset_info(args.dataset, args.remove_h)  # get configs for qm9 etc.

atom_encoder = dataset_info['atom_encoder']
atom_decoder = dataset_info['atom_decoder']

args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
dtype = torch.float32

if args.resume is not None:
    exp_name = args.exp_name + '_resume'
    start_epoch = args.start_epoch
    resume = args.resume  # resume this dir
    wandb_usr = args.wandb_usr
    normalization_factor = args.normalization_factor
    aggregation_method = args.aggregation_method
    mlp_type = args.mlp_type  # LEO

    with open(join('outputs', args.resume, 'args.pickle'), 'rb') as f:
        args = pickle.load(f)

    args.resume = resume
    args.break_train_epoch = False

    args.exp_name = exp_name  # new exp_name
    args.start_epoch = start_epoch
    args.wandb_usr = wandb_usr

    # Careful with this -->
    if not hasattr(args, 'normalization_factor'):
        args.normalization_factor = normalization_factor
    if not hasattr(args, 'aggregation_method'):
        args.aggregation_method = aggregation_method
    if not hasattr(args, 'mlp_type'):
        print("mlp_type is not found!")
        args.mlp_type = mlp_type

    print(args)

utils.create_folders(args)


# Wandb config
if args.no_wandb:
    mode = 'disabled'
else:
    mode = 'online' if args.online else 'offline'
#kwargs = {'entity': args.wandb_usr, 'name': args.exp_name, 'project': 'e3_diffusion', 'config': args,
#          'settings': wandb.Settings(_disable_stats=True), 'reinit': True, 'mode': mode}
kwargs = {'name': args.exp_name, 'project': 'e3_diffusion', 'config': args,
          'settings': wandb.Settings(_disable_stats=True), 'reinit': True, 'mode': mode}
wandb.init(**kwargs)
wandb.save('*.txt')

# Retrieve QM9 dataloaders
dataloaders, charge_scale = dataset.retrieve_dataloaders(args)

data_dummy = next(iter(dataloaders['train']))


if len(args.conditioning) > 0:
    print(f'Conditioning on {args.conditioning}')
    property_norms = compute_mean_mad(dataloaders, args.conditioning, args.dataset)  # compute mean, mad of each prop
    context_dummy = prepare_context(args.conditioning, data_dummy, property_norms)  # combines conditioning info into tensor
    context_node_nf = context_dummy.size(2)  # this is combined dim of all conditioning props
else:
    context_node_nf = 0
    property_norms = None

args.context_node_nf = context_node_nf

# Create EGNN flow
# vdm (with net), DistributionNodes (sample to get num of nodes), DistributionProperty (if conditioning)
# note that nodes_dist.sample is not filtered
model, nodes_dist, prop_dist = get_model(args, device, dataset_info, dataloaders['train'])
if prop_dist is not None:  # when conditioning
    prop_dist.set_normalizer(property_norms)
model = model.to(device)
optim = get_optim(args, model)
scheduler = get_scheduler(args, optim)
print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
if args.print_parameter_count:
    try:
        model.dynamics.print_parameter_count()
    except:
        print("Cannot print the number of model parameters")

gradnorm_queue = utils.Queue()  # stores grad norms for clipping within some std of past grads
gradnorm_queue.add(3000)  # Add large value that will be flushed.


def check_mask_correct(variables, node_mask):
    for variable in variables:
        if len(variable) > 0:
            assert_correctly_masked(variable, node_mask)


def main():
    if args.resume is not None:
        #flow_state_dict = torch.load(join('outputs', args.resume, 'flow.npy'))  # for vdm
        flow_state_dict = torch.load(join('outputs', args.resume, 'generative_model.npy'))  # for vdm
        optim_state_dict = torch.load(join('outputs', args.resume, 'optim.npy'))
        model.load_state_dict(flow_state_dict)
        optim.load_state_dict(optim_state_dict)
        if args.scheduler is not None:
            scheduler_state_dict = torch.load(join('outputs', args.resume, 'scheduler.npy'))
            scheduler.load_state_dict(scheduler_state_dict)

    # Initialize dataparallel if enabled and possible.
    if args.dp and torch.cuda.device_count() > 1:
        print(f'Training using {torch.cuda.device_count()} GPUs')
        model_dp = torch.nn.DataParallel(model.cpu())
        model_dp = model_dp.cuda()
    else:
        model_dp = model

    # Initialize model copy for exponential moving average of params.
    if args.ema_decay > 0:
        model_ema = copy.deepcopy(model)
        ema = flow_utils.EMA(args.ema_decay)

        # NOTE: LEO
        if args.resume is not None:
            ema_state_dict = torch.load(join('outputs', args.resume, 'generative_model_ema.npy'))
            model_ema.load_state_dict(ema_state_dict)

        if args.dp and torch.cuda.device_count() > 1:
            model_ema_dp = torch.nn.DataParallel(model_ema)  # used just for test
        else:
            model_ema_dp = model_ema
        
    else:
        ema = None
        model_ema = model
        model_ema_dp = model_dp  # how is this used?

    best_nll_val = 1e8
    best_nll_test = 1e8

    best_mol_stable = 0

    for epoch in range(args.start_epoch, args.n_epochs):
        wandb.log({"Epoch": epoch}, commit=True)
        print(f"--- Epoch {epoch} ---")

        if args.save_model_at_init:  # saves models in symdiff/outputs/exp_name on ziz
            utils.save_model(optim, 'outputs/%s/optim.npy' % args.exp_name)
            utils.save_model(model, 'outputs/%s/generative_model.npy' % args.exp_name)
            if scheduler is not None:
                utils.save_model(scheduler, 'outputs/%s/scheduler.npy' % args.exp_name)
            if args.ema_decay > 0:
                utils.save_model(model_ema, 'outputs/%s/generative_model_ema.npy' % args.exp_name)
            with open('outputs/%s/args.pickle' % args.exp_name, 'wb') as f:
                pickle.dump(args, f) 
            break

        #print("TEST WITH VAL FIRST")
        #nll_val = test(args=args, loader=dataloaders['valid'], epoch=epoch, eval_model=model,
        #           partition='Val', device=device, dtype=dtype, nodes_dist=nodes_dist,
        #           property_norms=property_norms)

        start_epoch = time.time()
        train_epoch(args=args, loader=dataloaders['train'], epoch=epoch, model=model, model_dp=model_dp,
                    model_ema=model_ema, ema=ema, device=device, dtype=dtype, property_norms=property_norms,
                    nodes_dist=nodes_dist, dataset_info=dataset_info,
                    gradnorm_queue=gradnorm_queue, optim=optim, scheduler=scheduler, prop_dist=prop_dist)
        print(f"Epoch took {time.time() - start_epoch:.1f} seconds.")

        if epoch % args.test_epochs == 0 and epoch != 0:  # NOTE: LEO
        #if epoch % args.test_epochs == 0:
            if isinstance(model, en_diffusion.EnVariationalDiffusion):
                if args.com_free:
                    wandb.log(model.log_info(), commit=True)  # should be constant for l2
            if not args.break_train_epoch:  # for debug
                # samples n_stability_samples points and compute atm_stable, mol_stable, validity, uniqueness and novelty
                validity_dict = analyze_and_save(args=args, epoch=epoch, model_sample=model_ema, nodes_dist=nodes_dist,
                                 dataset_info=dataset_info, device=device,
                                 prop_dist=prop_dist, n_samples=args.n_stability_samples)
                mol_stable = validity_dict["mol_stable"]
            # compute average nll over the val/test set
            nll_val = test(args=args, loader=dataloaders['valid'], epoch=epoch, eval_model=model_ema_dp,
                           partition='Val', device=device, dtype=dtype, nodes_dist=nodes_dist,
                           property_norms=property_norms)
            nll_test = test(args=args, loader=dataloaders['test'], epoch=epoch, eval_model=model_ema_dp,
                            partition='Test', device=device, dtype=dtype,
                            nodes_dist=nodes_dist, property_norms=property_norms)

            nll_val = nll_val[0]
            nll_test = nll_test[0]

            # add equivariance metrics here - TODO: remove
            if args.use_equivariance_metric:
                nll_val, model_metric_val, backbone_metric_val = nll_val
                nll_test, model_metric_test, backbone_metric_test = nll_test

            if nll_val < best_nll_val or args.force_save_model:  # NOTE: maybe also save best molecular stability?
                best_nll_val = nll_val
                best_nll_test = nll_test
                if args.save_model:  # saves models in symdiff/outputs/exp_name on ziz
                    args.current_epoch = epoch + 1
                    utils.save_model(optim, 'outputs/%s/optim.npy' % args.exp_name)
                    utils.save_model(model, 'outputs/%s/generative_model.npy' % args.exp_name)
                    if scheduler is not None:
                        utils.save_model(scheduler, 'outputs/%s/scheduler.npy' % args.exp_name)
                    if args.ema_decay > 0:
                        utils.save_model(model_ema, 'outputs/%s/generative_model_ema.npy' % args.exp_name)
                    with open('outputs/%s/args.pickle' % args.exp_name, 'wb') as f:
                        pickle.dump(args, f)

            # NOTE: added to save best model on mol stable
            if mol_stable > best_mol_stable or args.force_save_model:
                if args.save_model:  # saves models in symdiff/outputs/exp_name on ziz
                    utils.save_model(optim, 'outputs/%s/optim_ms.npy' % args.exp_name)
                    utils.save_model(model, 'outputs/%s/generative_model_ms.npy' % args.exp_name)
                    if scheduler is not None:
                        utils.save_model(scheduler, 'outputs/%s/scheduler_ms.npy' % args.exp_name)
                    if args.ema_decay > 0:
                        utils.save_model(model_ema, 'outputs/%s/generative_model_ema_ms.npy' % args.exp_name)
                    with open('outputs/%s/args_ms.pickle' % args.exp_name, 'wb') as f:
                        pickle.dump(args, f)

                # ???
                """
                if args.save_model:
                    utils.save_model(optim, 'outputs/%s/optim_%d.npy' % (args.exp_name, epoch))
                    utils.save_model(model, 'outputs/%s/generative_model_%d.npy' % (args.exp_name, epoch))
                    if args.ema_decay > 0:
                        utils.save_model(model_ema, 'outputs/%s/generative_model_ema_%d.npy' % (args.exp_name, epoch))
                    with open('outputs/%s/args_%d.pickle' % (args.exp_name, epoch), 'wb') as f:
                        pickle.dump(args, f)
                """
            print('Val loss: %.4f \t Test loss:  %.4f' % (nll_val, nll_test))
            print('Best val loss: %.4f \t Best test loss:  %.4f' % (best_nll_val, best_nll_test))
            wandb.log({"Val loss ": nll_val}, commit=True)
            wandb.log({"Test loss ": nll_test}, commit=True)
            wandb.log({"Best cross-validated test loss ": best_nll_test}, commit=True)

            if args.use_equivariance_metric:
                # finish this later
                wandb.log({"Val model equivariance metric ": model_metric_val}, commit=True)
                wandb.log({"Val backbone equivariance metric ": backbone_metric_val}, commit=True)
                wandb.log({"Test model equivariance metric ": model_metric_test}, commit=True)
                wandb.log({"Test backbone equivariance metric ": backbone_metric_test}, commit=True)

        # saves final model (by epochs)
        if epoch == args.n_epochs-1:
            if args.save_model:  # saves models in symdiff/outputs/exp_name on ziz
                utils.save_model(optim, 'outputs/%s/optim_final.npy' % args.exp_name)
                utils.save_model(model, 'outputs/%s/generative_model_final.npy' % args.exp_name)
                if scheduler is not None:
                    utils.save_model(scheduler, 'outputs/%s/scheduler_final.npy' % args.exp_name)
                if args.ema_decay > 0:
                    utils.save_model(model_ema, 'outputs/%s/generative_model_ema_final.npy' % args.exp_name)
                with open('outputs/%s/args_final.pickle' % args.exp_name, 'wb') as f:
                    pickle.dump(args, f)


if __name__ == "__main__":
    main()
