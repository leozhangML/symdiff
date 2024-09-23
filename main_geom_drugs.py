# Rdkit import should be first, do not move it
try:
    from rdkit import Chem
except ModuleNotFoundError:
    pass
import build_geom_dataset
from configs.datasets_config import geom_with_h

import copy
import utils
import argparse
import wandb
import torch
import time
import pickle

from os.path import join
from qm9.models import get_optim, get_scheduler, get_model

from equivariant_diffusion import en_diffusion
from equivariant_diffusion.utils import assert_correctly_masked
from equivariant_diffusion import utils as diffusion_utils

from qm9.utils import prepare_context, compute_mean_mad
import train_test
from train_test import train_epoch, test, analyze_and_save



################## General arguments ##################
parser = argparse.ArgumentParser(description='e3_diffusion')
parser.add_argument('--exp_name', type=str, default='debug_10')
parser.add_argument('--model', type=str, default='egnn_dynamics',
                    help='our_dynamics | schnet | simple_dynamics | '
                         'kernel_dynamics | egnn_dynamics |gnn_dynamics')
parser.add_argument('--probabilistic_model', type=str, default='diffusion',
                    help='diffusion')
parser.add_argument('--break_train_epoch', type=eval, default=False,
                    help='True | False')

################## Data arguments ##################
parser.add_argument('--datadir', type=str, default=None,
                    help='Give the path to the directory containing the data')


################## DiTGaussian args ################## 
parser.add_argument("--K", type=int, default=128, help="config for DiTGaussian")
parser.add_argument("--mlp_type", type=str, default="mlp", help="Config for DiTGaussian")                    


################## DiT_GNN and DiT_DiT args ##################
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


################## DiTEmb args ##################
parser.add_argument("--xh_hidden_size", type=int, default=128, help="config for DiTEmb")
parser.add_argument("--mlp_dropout", type=float, default=0.0, help="config for DiTEmb")


################## DiT args ##################
parser.add_argument("--out_channels", type=int, default=9, help="config for DiT")
parser.add_argument("--x_scale", type=float, default=25.0, help="config for DiT")
parser.add_argument("--hidden_size", type=int, default=256, help="config for DiT")
parser.add_argument("--depth", type=int, default=6, help="config for DiT")
parser.add_argument("--num_heads", type=int, default=4, help="config for DiT")
parser.add_argument("--mlp_ratio", type=float, default=2.0, help="config for DiT")
parser.add_argument("--subtract_x_0", action="store_true", help="config for DiT")


################## Sym_diff args ##################
parser.add_argument("--sigma_data", type=float, default=1.5, help="for VE scaling of inputs")
parser.add_argument("--com_free", action="store_true", help="whether to use the CoM subspace")
parser.add_argument("--rho", type=float, default=3, help="VE schedule")
parser.add_argument("--sigma_min", type=float, default=1e-5, help="VE schedule")
parser.add_argument("--sigma_max", type=float, default=10, help="VE schedule")

parser.add_argument("--print_grad_norms", action="store_true", help="whether to show the gamma and k grad norms")
parser.add_argument("--print_parameter_count", action="store_true", help="whether to show the gamma and k param count")


################## Model arguments ##################
# Training complexity is O(1) (unaffected), but sampling complexity O(steps).
parser.add_argument('--diffusion_steps', type=int, default=500)
parser.add_argument('--diffusion_noise_schedule', type=str, default='polynomial_2',
                    help='learned, cosine')
parser.add_argument('--diffusion_loss_type', type=str, default='l2',
                    help='vlb, l2')
parser.add_argument('--diffusion_noise_precision', type=float, default=1e-5)


# Positional embeddings
# Positional embeddings
parser.add_argument("--use_separate_gauss_embs", action="store_true", help="Whether to use separate Gaussian embeddings for k and gamma")
parser.add_argument("--gamma_K", type=float, default=0, help="K for gamma positional embeddings")
parser.add_argument("--k_K", type=float, default=0, help="K for k positional embeddings")
parser.add_argument("--pos_emb_gamma_projection_dim", type=float, default=0, help="Dimension of the projection for gamma positional embeddings")


# Use separate optimizers for K and Gamma
parser.add_argument("--use_separate_optimisers", action="store_true", help="Whether to use two separate optimizers for the K and Gamma", default=False)

parser.add_argument('--lr_K', type=float, default=2e-4)
parser.add_argument('--use_amsgrad_K', action="store_true")  # default from EDM
parser.add_argument('--weight_decay_K', type=float, default=1e-12)  # default from EDM

parser.add_argument('--lr_gamma', type=float, default=2e-4)
parser.add_argument('--use_amsgrad_gamma', action="store_true")  # default from EDM
parser.add_argument('--weight_decay_gamma', type=float, default=1e-12)  # default from EDM


# Use separate ema
parser.add_argument('--use_separate_ema', action="store_true", help='Use separate ema for the gamma and k', default=False)
parser.add_argument('--ema_decay_K', type=float, default=0.999,
                    help='Amount of EMA decay, 0 means off. A reasonable value'
                         ' is 0.999.')
parser.add_argument('--ema_decay_gamma', type=float, default=0.999,
                    help='Amount of EMA decay, 0 means off. A reasonable value'
                         ' is 0.999.')


# Seprate arguments for dropout        
parser.add_argument("--use_separate_dropout", action="store_true", help="Whether to use separate dropouts for gamma enc, gamma dec, and k")
parser.add_argument("--dropout_gamma_enc", type=float, default=0.0, help="Dropout for gamma encoder")
parser.add_argument("--dropout_gamma_dec", type=float, default=0.0, help="Dropout for gamma decoder")
parser.add_argument("--dropout_k", type=float, default=0.0, help="Dropout for k")

################## Optimization arguments ##################
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

parser.add_argument('--dp', type=eval, default=True,
                    help='True | False')
parser.add_argument('--condition_time', type=eval, default=True,
                    help='True | False')
parser.add_argument('--clip_grad', type=eval, default=True,
                    help='True | False')
parser.add_argument('--trace', type=str, default='hutch',
                    help='hutch | exact')


### Model parts to freeze for DIT DIT model  ####
parser.add_argument("--freeze_model_parts", action="store_true", help="Whether to freeze the model parts")
parser.add_argument("--model_part_to_freeze", type=str, default="", help="Which part of the model to freeze")


################## EGNN args ##################
parser.add_argument('--n_layers', type=int, default=6,
                    help='number of layers')
parser.add_argument('--inv_sublayers', type=int, default=1,
                    help='number of layers')
parser.add_argument('--nf', type=int, default=192,
                    help='number of layers')
parser.add_argument('--tanh', type=eval, default=True,
                    help='use tanh in the coord_mlp')
parser.add_argument('--attention', type=eval, default=True,
                    help='use attention in the EGNN')
parser.add_argument('--norm_constant', type=float, default=1,
                    help='diff/(|diff| + norm_constant)')
parser.add_argument('--sin_embedding', type=eval, default=False,
                    help='whether using or not the sin embedding')
################## EGNN args ##################
parser.add_argument('--ode_regularization', type=float, default=1e-3)
parser.add_argument('--dataset', type=str, default='geom',
                    help='dataset name')
parser.add_argument('--filter_n_atoms', type=int, default=None,
                    help='When set to an integer value, QM9 will only contain molecules of that amount of atoms')
parser.add_argument('--dequantization', type=str, default='argmax_variational',
                    help='uniform | variational | argmax_variational | deterministic')
parser.add_argument('--n_report_steps', type=int, default=50)
parser.add_argument('--wandb_usr', type=str)
parser.add_argument('--no_wandb', action='store_true', help='Disable wandb')
parser.add_argument('--online', type=bool, default=True, help='True = wandb online -- False = wandb offline')
parser.add_argument('--no-cuda', action='store_true', default=False, help='disable CUDA training')
parser.add_argument('--save_model', type=eval, default=True, help='save model')
parser.add_argument('--generate_epochs', type=int, default=1)
parser.add_argument('--num_workers', type=int, default=0,
                    help='Number of worker for the dataloader')
parser.add_argument('--test_epochs', type=int, default=1)
parser.add_argument('--data_augmentation', type=eval, default=False,
                    help='use attention in the EGNN')
parser.add_argument("--conditioning", nargs='+', default=[],
                    help='multiple arguments can be passed, '
                         'including: homo | onehot | lumo | num_atoms | etc. '
                         'usage: "--conditioning H_thermo homo onehot H_thermo"')
parser.add_argument('--resume', type=str, default=None,
                    help='')
parser.add_argument('--start_epoch', type=int, default=0,
                    help='')
parser.add_argument('--ema_decay', type=float, default=0,           # TODO
                    help='Amount of EMA decay, 0 means off. A reasonable value'
                         ' is 0.999.')
parser.add_argument('--augment_noise', type=float, default=0)
parser.add_argument('--n_stability_samples', type=int, default=20,
                    help='Number of samples to compute the stability')
parser.add_argument('--normalize_factors', type=eval, default=[1, 4, 10],
                    help='normalize factors for [x, categorical, integer]')

parser.add_argument('--remove_h', action='store_true')
parser.add_argument('--include_charges', type=eval, default=False, help='include atom charge or not')
parser.add_argument('--visualize_every_batch', type=int, default=5000)
parser.add_argument('--normalization_factor', type=float,
                    default=100, help="Normalize the sum aggregation of EGNN")
parser.add_argument('--aggregation_method', type=str, default='sum',
                    help='"sum" or "mean" aggregation for the graph network')
parser.add_argument('--filter_molecule_size', type=int, default=None,
                    help="Only use molecules below this size.")
parser.add_argument('--sequential', action='store_true',
                    help='Organize data by size to reduce average memory usage.')     # This is for GNNs, keep it off


parser.add_argument("--data_aug_at_sampling", action="store_true", help="Whether to augment data at sampling time")
parser.add_argument("--use_equivariance_metric", action="store_true", help="whether to log the equivariance metrics")
parser.add_argument("--n_importance_samples", type=int, default=32, help="whether to log the equivariance metrics")
parser.add_argument('--n_dims', type=int, default=3)

args = parser.parse_args()



##########################################################################################################################


# Get device
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
dtype = torch.float32

# Getting the dataset
data_file = args.datadir   #'./data/geom/geom_drugs_30.npy'                         
if args.remove_h:
    raise NotImplementedError('Remove H not implemented.')
else:
    dataset_info = geom_with_h

# Get the atom encoder and decoder - this converts the atom type to an integer and vice versa
atom_encoder = dataset_info['atom_encoder']
atom_decoder = dataset_info['atom_decoder']

# Split the data and apply transformations
split_data = build_geom_dataset.load_split_data(data_file, val_proportion=0.1, test_proportion=0.1, filter_size=args.filter_molecule_size)
transform = build_geom_dataset.GeomDrugsTransform(dataset_info, args.include_charges, device, args.sequential)

# Create dataloaders
dataloaders = {}
for key, data_list in zip(['train', 'val', 'test'], split_data):
    dataset = build_geom_dataset.GeomDrugsDataset(data_list, transform=transform)
    shuffle = (key == 'train') and not args.sequential

    # Sequential dataloading disabled for now.
    dataloaders[key] = build_geom_dataset.GeomDrugsDataLoader(
        sequential=args.sequential, dataset=dataset, batch_size=args.batch_size,
        shuffle=shuffle)
del split_data


# Get wandb arguments
# args, unparsed_args = parser.parse_known_args()
args.wandb_usr = utils.get_wandb_username(args.wandb_usr)

# If resume training, load the arguments from the previous run.
if args.resume is not None:
    # Get the arguments from the previous run
    exp_name = args.exp_name + '_resume'
    start_epoch = args.start_epoch
    resume = args.resume
    wandb_usr = args.wandb_usr
    mlp_type = args.mlp_type  # MLP type

    # Load the arguments from the previous run
    # with open(join(args.resume, 'args.pickle'), 'rb') as f:
    #     args = pickle.load(f)

    # Update the arguments
    args.resume = resume
    args.break_train_epoch = False
    args.exp_name = exp_name  # add _resume to the name
    args.start_epoch = start_epoch
    args.wandb_usr = wandb_usr
    print(args)

    # Get the model type
    if not hasattr(args, 'mlp_type'):
        print("mlp_type is not found!")
        args.mlp_type = mlp_type    

# Create folders
utils.create_folders(args)
print(args)


# Start wandb
if args.no_wandb:
    mode = 'disabled'
else:
    mode = 'online' if args.online else 'offline'
kwargs = {'name': args.exp_name, 'project': 'e3_diffusion_geom_drugs', 'config': args,
          'settings': wandb.Settings(_disable_stats=True), 'reinit': True, 'mode': mode}
wandb.init(**kwargs)
wandb.save('*.txt')

data_dummy = next(iter(dataloaders['train']))


# For conditional generation - ignore
if len(args.conditioning) > 0:
    print(f'Conditioning on {args.conditioning}')
    property_norms = compute_mean_mad(dataloaders, args.conditioning)
    context_dummy = prepare_context(args.conditioning, data_dummy, property_norms)
    context_node_nf = context_dummy.size(2)
else:
    context_node_nf = 0
    property_norms = None

args.context_node_nf = context_node_nf


# Create EGNN flow
model, nodes_dist, prop_dist = get_model(args, device, dataset_info, dataloader_train=dataloaders['train'])
model = model.to(device)
optim = get_optim(args, model)
scheduler = get_scheduler(args, optim)
print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
if args.print_parameter_count:
    model.dynamics.print_parameter_count()

# Set up gradient norm queue
gradnorm_queue = utils.Queue()
gradnorm_queue.add(3000)        # add large value that will be flushed.



def check_mask_correct(variables, node_mask):
    """ Check that the node mask is correctly applied to the variables.

    Args:
        variables (list): List of variables to check.
        node_mask (torch.Tensor):s Node mask to check against.
    """
    for variable in variables:
        if len(variable) > 0:
            assert_correctly_masked(variable, node_mask)


def main():
    # If resuming, load the model and optimizer state dicts.
    if args.resume is not None:
        flow_state_dict = torch.load(join(args.resume, 'generative_model.npy'))
        # dequantizer_state_dict = torch.load(join(args.resume, 'dequantizer.npy'))
        optim_state_dict = torch.load(join(args.resume, 'optim.npy'))
        model.load_state_dict(flow_state_dict)
        optim.load_state_dict(optim_state_dict)

        if args.scheduler is not None:
            scheduler_state_dict = torch.load(join(args.resume, 'scheduler.npy'))
            scheduler.load_state_dict(scheduler_state_dict)        

    # Initialize dataparallel if enabled and possible.
    if args.dp and torch.cuda.device_count() > 1 and args.cuda:
        print(f'Training using {torch.cuda.device_count()} GPUs')
        model_dp = torch.nn.DataParallel(model.cpu())
        model_dp = model_dp.cuda()
    else:
        model_dp = model

    # Initialize model copy for exponential moving average of params.
    if args.ema_decay > 0:
        model_ema = copy.deepcopy(model)
        ema = diffusion_utils.EMA(args.ema_decay)

        if args.resume is not None:
            ema_state_dict = torch.load(join(args.resume, 'generative_model_ema.npy'))
            model_ema.load_state_dict(ema_state_dict)        

        if args.dp and torch.cuda.device_count() > 1:
            model_ema_dp = torch.nn.DataParallel(model_ema)
        else:
            model_ema_dp = model_ema
    else:
        ema = None
        model_ema = model
        model_ema_dp = model_dp

    # Initialize values for best nll
    best_nll_val = 1e8
    best_nll_test = 1e8
    best_mol_stable = 0

    # Start training
    for epoch in range(args.start_epoch, args.n_epochs):
        wandb.log({"Epoch": epoch}, commit=True)
        print(f"--- Epoch {epoch} ---")        
        start_epoch = time.time()

        # Train for one epoch
        train_epoch(args=args, loader=dataloaders['train'], epoch=epoch, model=model, model_dp=model_dp,
                    model_ema=model_ema, ema=ema, device=device, dtype=dtype, property_norms=property_norms,
                    nodes_dist=nodes_dist, dataset_info=dataset_info,
                    gradnorm_queue=gradnorm_queue, optim=optim, scheduler=scheduler, prop_dist=prop_dist)
        print(f"Epoch took {time.time() - start_epoch:.1f} seconds.")

        # Check if we should test the model (last, first, or every test_epochs)
        if epoch % args.test_epochs == 0 or epoch == args.n_epochs - 1 or epoch == start_epoch: 
            if isinstance(model, en_diffusion.EnVariationalDiffusion):
                if args.com_free:
                    wandb.log(model.log_info(), commit=True)  # should be constant for l2                

            # Test for molecular metrics
            if not args.break_train_epoch:
                validity_dict = analyze_and_save(args=args, epoch=epoch, model_sample=model_ema, nodes_dist=nodes_dist,
                                                dataset_info=dataset_info, device=device,
                                                prop_dist=prop_dist, n_samples=args.n_stability_samples)
                mol_stable = validity_dict["mol_stable"]                                                

            # Test for NLL
            nll_val, _, _ = test(args=args, loader=dataloaders['val'], epoch=epoch, eval_model=model_ema_dp,
                           partition='Val', device=device, dtype=dtype, nodes_dist=nodes_dist,
                           property_norms=property_norms)
            nll_test, _, _ = test(args=args, loader=dataloaders['test'], epoch=epoch, eval_model=model_ema_dp,
                            partition='Test', device=device, dtype=dtype,
                            nodes_dist=nodes_dist, property_norms=property_norms)


            # Save the model if it is the best model
            if nll_val < best_nll_val:
                best_nll_val = nll_val
                best_nll_test = nll_test
                if args.save_model:
                    args.current_epoch = epoch + 1
                    utils.save_model(optim, 'outputs/%s/optim.npy' % args.exp_name)
                    utils.save_model(model, 'outputs/%s/generative_model.npy' % args.exp_name)
                    if scheduler is not None:
                        utils.save_model(scheduler, 'outputs/%s/scheduler.npy' % args.exp_name)                    
                    if args.ema_decay > 0:
                        utils.save_model(model_ema, 'outputs/%s/generative_model_ema.npy' % args.exp_name)
                    with open('outputs/%s/args.pickle' % args.exp_name, 'wb') as f:
                        pickle.dump(args, f)

            # Save model with the best moleculular stability
            if mol_stable > best_mol_stable:
                if args.save_model:                         # saves models in symdiff/outputs/exp_name on ziz
                    utils.save_model(optim, 'outputs/%s/optim_ms.npy' % args.exp_name)
                    utils.save_model(model, 'outputs/%s/generative_model_ms.npy' % args.exp_name)
                    if scheduler is not None:
                        utils.save_model(scheduler, 'outputs/%s/scheduler_ms.npy' % args.exp_name)
                    if args.ema_decay > 0:
                        utils.save_model(model_ema, 'outputs/%s/generative_model_ema_ms.npy' % args.exp_name)
                    with open('outputs/%s/args_ms.pickle' % args.exp_name, 'wb') as f:
                        pickle.dump(args, f)            

            print('Val loss: %.4f \t Test loss:  %.4f' % (nll_val, nll_test))
            print('Best val loss: %.4f \t Best test loss:  %.4f' % (best_nll_val, best_nll_test))
            wandb.log({"Val loss ": nll_val}, commit=True)
            wandb.log({"Test loss ": nll_test}, commit=True)
            wandb.log({"Best cross-validated test loss ": best_nll_test}, commit=True)


if __name__ == "__main__":
    main()
