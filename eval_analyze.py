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
import numpy as np
import json


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


def test(args, flow_dp, nodes_dist, device, dtype, loader, partition='Test', num_passes=1,
         eval_indices = None, eval_times = None):
    flow_dp.eval()
    nll_epoch = 0
    n_samples = 0

    # Get args
    save_nlls = args.save_nlls_for_loaders
    if partition == "Test":
        gamma_on_indices = args.gamma_on_indices
        n_gamma_samples_for_x = args.n_gamma_samples_for_x
    else:
        gamma_on_indices = False
        n_gamma_samples_for_x = 0

    # Init
    if gamma_on_indices:
        gammas_for_indices = {idx: [] for idx in eval_indices}
    all_nlls = []
    all_ts = []
    current_idx = 0    

    # Get results
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

                # COM free
                x = remove_mean_with_mask(x, node_mask)
                check_mask_correct([x, one_hot], node_mask)
                assert_mean_zero_with_mask(x, node_mask)
                h = {'categorical': one_hot, 'integer': charges}

                # Changes for conditioning
                if len(args.conditioning) > 0:
                    context = prepare_context(args.conditioning, data).to(device, dtype)
                    assert_correctly_masked(context, node_mask)
                else:
                    context = None


                # Subset the data if we are evaluating for indices
                if gamma_on_indices:
                    # Check if any of our indices are in the current batch
                    print("The current batch indices are", np.arange(current_idx, current_idx + batch_size))
                    indices_in_batch = np.isin(np.arange(current_idx, current_idx + batch_size), eval_indices)
                    eval_indices_in_batch = np.isin(eval_indices, np.arange(current_idx, current_idx + batch_size))
                    if np.any(indices_in_batch):
                        # Get the indices in the batch
                        indices_in_batch = np.where(indices_in_batch)[0]
                        eval_indices_in_batch = np.where(eval_indices_in_batch)[0]
                        ts_in_batch = eval_times[eval_indices_in_batch]
                        print("Indices in batch and current index", indices_in_batch, current_idx)
                        print("Eval indices in batch", eval_indices_in_batch)   

                        # Reshape ts so that it is of shape (, 1) and so that it is on device
                        ts_in_batch = ts_in_batch.reshape(-1, 1)
                        ts_in_batch = torch.tensor(ts_in_batch, dtype=dtype, device=device)
                        print("Shape of ts in batch", ts_in_batch.shape)                        

                        # Subset our data 
                        x = x[indices_in_batch]
                        h = {key: h[key][indices_in_batch] for key in h}
                        node_mask = node_mask[indices_in_batch]
                        edge_mask = edge_mask[indices_in_batch]
                        if context is not None:
                            context = context[indices_in_batch]

                        # Print the shapes of x, h, node_mask and edge_mask
                        print("Shape of x", x.shape)
                        print("Shape of h", h['categorical'].shape)
                        print("Shape of node_mask", node_mask.shape)
                        print("Shape of edge_mask", edge_mask.shape)                                

                    else:
                        # If none of the indices are in the batch, continue to the next batch
                        current_idx += batch_size
                        continue


                # Transform batch through flow
                if save_nlls:
                    nll, _, _, loss_dict = losses.compute_loss_and_nll(args, flow_dp, nodes_dist, x, h, node_mask,
                                                            edge_mask, context, return_time=True)                 
                else:
                    nll, _, _ = losses.compute_loss_and_nll(args, flow_dp, nodes_dist, x, h, node_mask,
                                                            edge_mask, context)

                # Get samples for gammas 
                if gamma_on_indices: 
                    gamma_samples = []
                    for i in range(n_gamma_samples_for_x):
                        gammas = losses.sample_gammas(args, flow_dp, nodes_dist, x, h, node_mask, edge_mask, context, t_fixed=ts_in_batch)
                        gamma_samples.append(gammas)

                    # Add the samples to the dictionary
                    for idx, gammas in zip(indices_in_batch, gamma_samples):
                        gammas_for_indices[current_idx + idx].append(gammas.detach().cpu().numpy())


                # Flatten the vector of NLLs and add them to the list of all NLLs
                print("Shape of nll in main", nll.shape)
                nll = nll.reshape(-1)
                print("Shape of nll in main after rehspaing", nll.shape)

                if save_nlls:
                    all_nlls.append(nll.detach().cpu().numpy())
                    all_ts.append(loss_dict['t'].detach().cpu().numpy())
        

                nll_epoch += nll.item() * batch_size
                n_samples += batch_size
                if i % args.n_report_steps == 0:
                    print(f"\r {partition} NLL \t, iter: {i}/{len(loader)}, "
                          f"NLL: {nll_epoch/n_samples:.2f}")
                    
                                # After the current index
                current_idx += batch_size


    # If save_nlls return all nlls for the loader as well as the mean nll
    if save_nlls:
        return nll_epoch/n_samples, all_nlls, all_ts                
    if gamma_on_indices:
        return gammas_for_indices
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
    parser.add_argument('--use_gamma_for_sampling', type=bool, default=True)

    # New arguments

    # Arguments for finding NLLs
    parser.add_argument('--save_nlls_for_loaders', type=bool, default=False)
    parser.add_argument('--find_top_nlls', type=bool, default=False)    
    parser.add_argument('--top_k_nlls', type=int, default=10)
    parser.add_argument("--path_to_save_top_nll_indices", type=str, default=None)        

    # Arguments for sampling gamma on particular data indices
    parser.add_argument('--gamma_on_indices', type=bool, default=False)
    parser.add_argument("--gamma_indices_path", type=str, default=None)     
    parser.add_argument("--gamma_times_path", type=str, default=None)   
    parser.add_argument('--n_gamma_samples_for_x', type=int, default=1000)
    parser.add_argument("--gamma_samples_save_path", type=str, default=None)        
    parser.add_argument('--only_find_gammas', type=bool, default=False)    


    eval_args, unparsed_args = parser.parse_known_args()
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

    # Add argument for gamma sampling
    args.use_gamma_for_sampling = eval_args.use_gamma_for_sampling
    print("Value for gamma sampling", args.use_gamma_for_sampling)

    # Arguments for saving nlls
    args.save_nlls_for_loaders = eval_args.save_nlls_for_loaders
    args.find_top_nlls = eval_args.find_top_nlls
    args.top_k_nlls = eval_args.top_k_nlls
    args.path_to_save_top_nll_indices = eval_args.path_to_save_top_nll_indices

    # Arguments for sampling gamma on particular data indices
    args.gamma_on_indices = eval_args.gamma_on_indices
    args.gamma_indices_path = eval_args.gamma_indices_path
    args.gamma_times_path = eval_args.gamma_times_path
    args.n_gamma_samples_for_x = eval_args.n_gamma_samples_for_x
    args.gamma_samples_save_path = eval_args.gamma_samples_save_path
    args.only_find_gammas = eval_args.only_find_gammas

    # Add other arguments
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")
    args.device = device
    dtype = torch.float32
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

    fn = 'generative_model_ema.npy' if args.ema_decay > 0 or args.use_separate_emas else 'generative_model.npy'
    flow_state_dict = torch.load(join(eval_args.model_path, fn), map_location=device)
    generative_model.load_state_dict(flow_state_dict)

    # Analyze stability, validity, uniqueness and novelty with nodes_dist
    stability_dict = None
    rdkit_metrics = None
    if not args.only_find_gammas:
        stability_dict, rdkit_metrics = analyze_and_save(
            args, eval_args, device, generative_model, nodes_dist,
            prop_dist, dataset_info, n_samples=eval_args.n_samples,
            batch_size=eval_args.batch_size_gen, save_to_xyz=eval_args.save_to_xyz)
        print(stability_dict)

        if rdkit_metrics is not None:
            rdkit_metrics = rdkit_metrics[0]
            print("Validity %.4f, Uniqueness: %.4f, Novelty: %.4f" % (rdkit_metrics[0], rdkit_metrics[1], rdkit_metrics[2]))
        else:
            print("Install rdkit roolkit to obtain Validity, Uniqueness, Novelty")


    # In GEOM-Drugs the validation partition is named 'val', not 'valid'.
    if args.dataset == 'geom':
        val_name = 'val'
        num_passes = 1
    else:
        val_name = 'valid'
        num_passes = 5

    # Evaluate negative log-likelihood for the validation and test partitions
    if not args.save_nlls_for_loaders and not args.only_find_gammas:
        val_nll = test(args, generative_model, nodes_dist, device, dtype,
                    dataloaders[val_name],
                    partition='Val')
        print(f'Final val nll {val_nll}')


    # Test NLL
    if args.save_nlls_for_loaders:
        test_nll, all_test_nlls, all_test_ts = test(args, generative_model, nodes_dist, device, dtype,
                                                    dataloaders['test'],
                                                    partition='Test', num_passes=num_passes)    
        
    elif args.only_find_gammas:
        indices = np.load(args.gamma_indices_path)
        times = np.load(args.gamma_times_path)
        print("Shape of indices and times", indices.shape, times.shape)

        times = torch.tensor(times)
        print("Shape of times after conversion to torch", times.shape)

        # Get gammas and NLLs
        gammas_for_indices = test(args, generative_model, nodes_dist, device, dtype,
                                                dataloaders['test'], partition='Test', num_passes=num_passes, 
                                                eval_indices=indices, eval_times=times)

    else:
        test_nll = test(args, generative_model, nodes_dist, device, dtype,
                        dataloaders['test'],
                        partition='Test', num_passes=num_passes)
        print(f'Final test nll {test_nll}')

    
    # If finding gammas, save everything
    if args.only_find_gammas:
        print("Saving gammas")
        # Check if the folder for the eval_indices_save path exists and if it does not create it
        if not os.path.exists(os.path.dirname(args.gamma_samples_save_path)):
            os.makedirs(os.path.dirname(args.gamma_samples_save_path))
            print(f"Created directory {os.path.dirname(args.gamma_samples_save_path)}")

        # Save the gammas in a simialr way 
        for k, v in gammas_for_indices.items():
            gammas_for_indices[k] = [val for val in v]

        # Save the dictionary to the path as a json file
        save_path = f"{args.gamma_samples_save_path}/gammas_for_indices.json"
        with open(save_path, 'w') as f:
            json.dump(gammas_for_indices, f)            



    # Save
    if not args.save_nlls_for_loaders and not args.only_find_gammas:
        print(f'Overview: val nll {val_nll} test nll {test_nll}', stability_dict)
    else:
        print(f'Overview: Test nll {test_nll}', stability_dict)

    with open(join(eval_args.model_path, 'eval_log.txt'), 'w') as f:
        if not args.save_nlls_for_loaders:
            print(f'Overview: val nll {val_nll} test nll {test_nll}',
                stability_dict,
                file=f)
        else:
            print(f'Overview: Test nll {test_nll}', stability_dict, file=f)            


    # Saving the NLLS for the loaders as numpy arrays
    if args.save_nlls_for_loaders:
        all_test_nlls = np.concatenate(all_test_nlls)
        print("Shape of all nlls after concatenation", all_test_nlls.shape)
        np.save(join(eval_args.model_path, f'all_test_nlls.npy'), all_test_nlls)
        print(f"Saved all nlls to {join(eval_args.model_path, f'all_test_nlls.npy')}")

        all_test_ts = np.concatenate(all_test_ts)
        print("Shape of all ts after concatenation", all_test_ts.shape)
        np.save(join(eval_args.model_path, f'all_test_ts.npy'), all_test_ts)
        print(f"Saved all ts to {join(eval_args.model_path, f'all_test_ts.npy')}")


    # Get the top and bottom k NLLs
    if args.find_top_nlls:
        # Get the top K and bottom k for the test nlls
        print("Getting top k and bottom k NLLs")
        top_k_indices = np.argsort(all_test_nlls)[:args.top_k_nlls]
        top_k_corresponding_times = all_test_ts[top_k_indices]
        top_k_nlls = all_test_nlls[top_k_indices]
        print("Top k indices", top_k_indices)

        bottom_k_indices = np.argsort(all_test_nlls)[-args.top_k_nlls:]
        bottom_k_corresponding_times = all_test_ts[bottom_k_indices]
        bottom_k_nlls = all_test_nlls[bottom_k_indices]
        print("Bottom k indices", bottom_k_indices)

        # Save the indices as a numpy array in the file path that is specified in the configs
        if not os.path.exists(args.path_to_save_top_nll_indices):
            os.makedirs(args.path_to_save_top_nll_indices)
            print(f"Created directory {args.path_to_save_top_nll_indices}")

        # Save
        np.save(f"{args.path_to_save_top_nll_indices}/top_k_indices.npy", top_k_indices)
        print(f"Saved top k indices to {args.path_to_save_top_nll_indices}/top_k_indices.npy")
        np.save(f"{args.path_to_save_top_nll_indices}/top_k_times.npy", top_k_corresponding_times)
        print(f"Saved top k times to {args.path_to_save_top_nll_indices}/top_k_times.npy")
        np.save(f"{args.path_to_save_top_nll_indices}/top_k_nlls.npy", top_k_nlls)
        print(f"Saved top k nlls to {args.path_to_save_top_nll_indices}/top_k_nlls.npy")
        

if __name__ == "__main__":
    main()
