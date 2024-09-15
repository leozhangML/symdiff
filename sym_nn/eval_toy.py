import argparse
import os
import pickle

import torch

from qm9.models import get_model

def main():
    
    parser = argparse.ArgumentParser(description='eval_toy_experiment')
    parser.add_argument('--model_path', type=str, default="")

    eval_args = parser.parse_args()

    with open(os.path.join(eval_args.model_path, 'args.pickle'), 'rb') as f:
        args = pickle.load(f)

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")
    args.device = device
    dtype = torch.float32

    generative_model, _, _ = get_model(args, device, None, None)
    generative_model.to(device)

    fn = 'generative_model_ema.npy' if args.ema_decay > 0 else 'generative_model.npy'
    flow_state_dict = torch.load(os.path.join(eval_args.model_path, fn), map_location=device)
    generative_model.load_state_dict(flow_state_dict)

    # Evaluate negative log-likelihood for the validation and test partitions
    #val_nll = test(args, generative_model, None, device, dtype,
    #               dataloaders[val_name],
    #               partition='Val')
    #print(f'Final val nll {val_nll}')

if __name__ == "__main__":
    main()