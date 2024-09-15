from abc import ABC
import os

import torch
from torch.utils.data import DataLoader, Dataset

from sym_nn.utils import orthogonal_haar


class Distribution(ABC):

    def __init__(self):
        pass

    def sample(self, n):
        """
        Return dict of:
        positions: [n, 2, 2],
        atom_mask: [n, 2, 1],
        edge_mask: torch.empty(n),
        one_hot: [n, 2, 1]
        """
        raise NotImplementedError


class TestDistribution(Distribution):
    """Annulus distribution"""

    def __init__(self, args):
        self.min_radius = args.min_radius
        self.max_radius = args.max_radius

    def sample(self, n, rotate=False):

        radii = self.min_radius + torch.rand(n, 1) * (
            self.max_radius - self.min_radius
        )

        # [n, 2]
        if rotate:
            # sample uniform from circle 
            positions = radii * orthogonal_haar(
                dim=2, target_tensor=torch.empty(n)
                )[:, :, 0]
        else:
            positions = torch.cat(
                [radii, torch.zeros(n, 1)],
                dim=-1
                )

        # [n, 2, 2]
        positions = torch.cat(
            [positions.unsqueeze(1), -positions.unsqueeze(1)],
            dim=1
            )

        atom_mask = torch.ones(n, 2)
        edge_mask = torch.empty(n, 2, 2)
        one_hot = torch.zeros(n, 2, 1)

        return {"positions": positions, "atom_mask": atom_mask, 
                "edge_mask": edge_mask, "one_hot": one_hot}


class ProcessedDataset(Dataset):

    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.data.items()}

    def __len__(self):
        return len(self.data["positions"])
    

def retrieve_distribution(args):

    if args.dataset == "test":
        distribution = TestDistribution(args)
    else:
        ValueError

    return distribution


def retrieve_dataloaders(args):

    toy_dataset_path = os.path.join(args.toy_datasets_path, args.dataset)

    if os.path.exists(toy_dataset_path) and not args.resample_toy_data:
        train_dataset = torch.load(os.path.join(toy_dataset_path, "train.pt"))
        val_dataset = torch.load(os.path.join(toy_dataset_path, "valid.pt"))
        test_dataset = torch.load(os.path.join(toy_dataset_path, "test.pt"))
    else:
        distribution = retrieve_distribution(args)
        train_dataset = ProcessedDataset(distribution.sample(args.toy_train_n, rotate=args.toy_train_rotate))
        val_dataset = ProcessedDataset(distribution.sample(args.toy_val_n))
        test_dataset = ProcessedDataset(distribution.sample(args.toy_test_n))

        try:
            os.makedirs(toy_dataset_path)
        except OSError:
            pass

        torch.save(train_dataset, os.path.join(toy_dataset_path, "train.pt"))
        torch.save(val_dataset, os.path.join(toy_dataset_path, "valid.pt"))
        torch.save(test_dataset, os.path.join(toy_dataset_path, "test.pt"))

    datasets = {"train": train_dataset, "valid": val_dataset, "test": test_dataset}

    # standard collate should be fine?
    dataloaders = {split: DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True
    ) for split, dataset in datasets.items()}

    return dataloaders


