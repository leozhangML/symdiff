# Rdkit import should be first, do not move it
try:
    from rdkit import Chem
except ModuleNotFoundError:
    pass
import utils
import argparse
from qm9 import dataset
from qm9.models import get_model
import os
from equivariant_diffusion.utils import assert_mean_zero_with_mask, remove_mean_with_mask,\
    assert_correctly_masked
import torch
import time
import pickle
from configs.datasets_config import get_dataset_info
from os.path import join
from qm9.sampling import sample
from qm9.analyze import analyze_stability_for_molecules, analyze_node_distribution
from qm9.utils import prepare_context, compute_mean_mad
from qm9 import visualizer as qm9_visualizer
import qm9.losses as losses

from sym_nn.visualise_gamma import get_samples, sample_gamma

try:
    from qm9 import rdkit_functions
except ModuleNotFoundError:
    print('Not importing rdkit functions.')


def check_mask_correct(variables, node_mask):
    for variable in variables:
        assert_correctly_masked(variable, node_mask)


def analyze_and_save(args, eval_args, device, generative_model,
                     nodes_dist, prop_dist, dataset_info, n_samples=10,
                     batch_size=10, save_to_xyz=False):
    batch_size = min(batch_size, n_samples)
    assert n_samples % batch_size == 0
    molecules = {'one_hot': [], 'x': [], 'node_mask': []}
    start_time = time.time()
    for i in range(int(n_samples/batch_size)):
        nodesxsample = nodes_dist.sample(batch_size)  # NOTE: can't use this with filtered
        one_hot, charges, x, node_mask = sample(
            args, device, generative_model, dataset_info, prop_dist=prop_dist, nodesxsample=nodesxsample)

        molecules['one_hot'].append(one_hot.detach().cpu())
        molecules['x'].append(x.detach().cpu())
        molecules['node_mask'].append(node_mask.detach().cpu())

        current_num_samples = (i+1) * batch_size
        secs_per_sample = (time.time() - start_time) / current_num_samples
        print('\t %d/%d Molecules generated at %.2f secs/sample' % (
            current_num_samples, n_samples, secs_per_sample))

        if save_to_xyz:
            id_from = i * batch_size
            qm9_visualizer.save_xyz_file(
                join(eval_args.model_path, 'eval/analyzed_molecules/'),
                one_hot, charges, x, dataset_info, id_from, name='molecule',
                node_mask=node_mask)

    molecules = {key: torch.cat(molecules[key], dim=0) for key in molecules}  # what are shapes to be able to do this?
    stability_dict, rdkit_metrics = analyze_stability_for_molecules(
        molecules, dataset_info)

    return stability_dict, rdkit_metrics


def test(args, flow_dp, nodes_dist, device, dtype, loader, partition='Test', num_passes=1):
    flow_dp.eval()
    nll_epoch = 0
    n_samples = 0
    for pass_number in range(num_passes):
        with torch.no_grad():
            for i, data in enumerate(loader):
                # Get data
                x = data['positions'].to(device, dtype)
                node_mask = data['atom_mask'].to(device, dtype).unsqueeze(2)
                edge_mask = data['edge_mask'].to(device, dtype)
                one_hot = data['one_hot'].to(device, dtype)
                charges = (data['charges'] if args.include_charges else torch.zeros(0)).to(device, dtype)

                batch_size = x.size(0)

                x = remove_mean_with_mask(x, node_mask)
                check_mask_correct([x, one_hot], node_mask)
                assert_mean_zero_with_mask(x, node_mask)

                h = {'categorical': one_hot, 'integer': charges}

                if len(args.conditioning) > 0:
                    context = prepare_context(args.conditioning, data).to(device, dtype)
                    assert_correctly_masked(context, node_mask)
                else:
                    context = None

                # transform batch through flow
                nll, _, _ = losses.compute_loss_and_nll(args, flow_dp, nodes_dist, x, h, node_mask,
                                                        edge_mask, context)
                # standard nll from forward KL

                nll_epoch += nll.item() * batch_size
                n_samples += batch_size
                if i % args.n_report_steps == 0:
                    print(f"\r {partition} NLL \t, iter: {i}/{len(loader)}, "
                          f"NLL: {nll_epoch/n_samples:.2f}")

    print("The type of the output of test() is:", type(nll_epoch/n_samples))
    return nll_epoch/n_samples


def main():

    # compute stability, nll etc. metrics on val/test sets for model_path

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="outputs/edm_1",
                        help='Specify model path')
    parser.add_argument('--n_samples', type=int, default=100,
                        help='Specify model path')
    parser.add_argument('--batch_size_gen', type=int, default=100,
                        help='Specify model path')
    parser.add_argument('--save_to_xyz', type=eval, default=False,
                        help='Should save samples to xyz files.')
    parser.add_argument("--datadir", type=str, default=None, 
                        help="Use if trained on a different node")

    # options for what eval metrics to compute
    parser.add_argument("--compute_chemical_metrics", action="store_true")
    parser.add_argument("--compute_nll", action="store_true")
    parser.add_argument("--model_equivariance_metric", action="store_true")
    parser.add_argument("--return_iwae_nll", action="store_true")
    parser.add_argument('--n_importance_samples', type=int, default=10)
    parser.add_argument("--visualise_gamma", action="store_true")
    parser.add_argument('--plots_path', type=str, default="/data/ziz/not-backed-up/lezhang/projects/symdiff/plots")

    eval_args, unparsed_args = parser.parse_known_args()


    # load training args
    assert eval_args.model_path is not None

    with open(join(eval_args.model_path, 'args.pickle'), 'rb') as f:
        args = pickle.load(f)

    if eval_args.datadir is not None:
        print("Using different datadir!")
        args.datadir = eval_args.datadir


    # NOTE: CAREFUL with this -->
    if not hasattr(args, 'normalization_factor'):
        args.normalization_factor = 1
    if not hasattr(args, 'aggregation_method'):
        args.aggregation_method = 'sum'

    if not hasattr(args, 'mlp_type'):  # LEO
        print("mlp_type not found!")
        args.mlp_type = 'mlp'
    if not hasattr(args, 'rho'):  # LEO
        print("rho not found!")
        args.rho = None
    if not hasattr(args, 'sigma_min'):  # LEO
        print("sigma_min not found!")
        args.sigma_min = None
    if not hasattr(args, 'sigma_max'):  # LEO
        print("sigma_max not found!")
        args.sigma_max = None
    if not hasattr(args, 'molecule'):  # LEO
        print("molecule not found!")
        args.molecule = True
    if not hasattr(args, 'n_dims'):  # LEO
        print("n_dims not found!")
        args.n_dims = 3
    if not hasattr(args, 'com_free'):  # LEO
        print("com_free not found!")
        args.com_free = True

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")
    args.device = device
    dtype = torch.float32
    args.dtype = dtype
    utils.create_folders(args)
    print(args)

    # Retrieve QM9 dataloaders
    dataloaders, charge_scale = dataset.retrieve_dataloaders(args)
    dataset_info = get_dataset_info(args.dataset, args.remove_h)

    # Load model
    generative_model, nodes_dist, prop_dist = get_model(args, device, dataset_info, dataloaders['train'])
    if prop_dist is not None:
        property_norms = compute_mean_mad(dataloaders, args.conditioning, args.dataset)
        prop_dist.set_normalizer(property_norms)
    generative_model.to(device)
    print("Loaded model!")

    fn = 'generative_model_ema.npy' if args.ema_decay > 0 else 'generative_model.npy'
    flow_state_dict = torch.load(join(eval_args.model_path, fn), map_location=device)
    generative_model.load_state_dict(flow_state_dict)
    print("Loaded model state dict!")

    # Analyze stability, validity, uniqueness and novelty with nodes_dist
    if eval_args.compute_chemical_metrics:
        print("Computing chemical metrics")
        stability_dict, rdkit_metrics = analyze_and_save(
            args, eval_args, device, generative_model, nodes_dist,
            prop_dist, dataset_info, n_samples=eval_args.n_samples,
            batch_size=eval_args.batch_size_gen, save_to_xyz=eval_args.save_to_xyz)
        
        print("Stability dict after analysing:")
        print(stability_dict)

        if rdkit_metrics is not None:
            rdkit_metrics = rdkit_metrics[0]
            print("Validity %.4f, Uniqueness: %.4f, Novelty: %.4f" % (rdkit_metrics[0], rdkit_metrics[1], rdkit_metrics[2]))
        else:
            print("Install rdkit roolkit to obtain validity, uniqueness and novelty metrics")

    # In GEOM-Drugs the validation partition is named 'val', not 'valid'.
    if args.dataset == 'geom':
        val_name = 'val'
        num_passes = 1
    else:
        val_name = 'valid'
        num_passes = 5

    if eval_args.return_iwae_nll:
        args.return_iwae_nll = True
        args.n_importance_samples = eval_args.n_importance_samples

    if eval_args.model_equivariance_metric:
        args.model_equivariance_metric = True

    # Evaluate negative log-likelihood for the validation and test partitions
    if eval_args.compute_nll:
        print("Evaluating NLL")
        try:
            val_nll, val_model_metric, val_backbone_metric, val_iwae_nll = test(args, generative_model, nodes_dist, device, dtype,
                        dataloaders[val_name],
                        partition='Val')
        except:
            val_nll = test(args, generative_model, nodes_dist, device, dtype, 
                            dataloaders[val_name], partition='Val')
        print(f'Final val. NLL {val_nll}')

        try:
            test_nll, test_model_metric, test_backbone_metric, test_iwae_nll = test(args, generative_model, nodes_dist, device, dtype,
                            dataloaders['test'],
                            partition='Test', num_passes=num_passes)
        except:
            test_nll= test(args, generative_model, nodes_dist, device, dtype,
                            dataloaders['test'],
                            partition='Test', num_passes=num_passes)            
            
        # Print the, being careful of model metrics
        print(f'Final test NLL {test_nll}')
        print(f'Overview: val. NLL {val_nll} test NLL {test_nll}', stability_dict)
        # print(f"(Val) model metric: {val_model_metric}, backbone metric: {val_backbone_metric}, val_iwae_nll: {val_iwae_nll}")
        # print(f"(Test) model metric: {test_model_metric}, backbone metric: {test_backbone_metric}, test_iwae_nll: {test_iwae_nll}")

        # Saving results
        with open(join(eval_args.model_path, 'eval_log.txt'), 'w') as f:
            print("Saving in:", join(eval_args.model_path, 'eval_log.txt'))
            # print(f'Overview: val nll {val_nll}, test nll {test_nll}, (Val) model metric: {val_model_metric}, \
            #       backbone metric: {val_backbone_metric}, val_iwae_nll: {val_iwae_nll}, (Test) model metric: {test_model_metric}, \
            #       backbone metric: {test_backbone_metric}, test_iwae_nll: {test_iwae_nll}',
            #     stability_dict,
            #     file=f)
            print(f'Overview: val nll {val_nll}, test nll {test_nll}',
                stability_dict,
                file=f)
            

    # visualise gamma
    if eval_args.visualise_gamma:
        print("Visualising gamma samples:")
        z_t, t, node_mask = get_samples(args, generative_model, dataloaders['train'])
        sample_gamma(args, eval_args, generative_model, z_t, t, node_mask, num_samples=10000)


if __name__ == "__main__":
    main()
