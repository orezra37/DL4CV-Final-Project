from pathlib import Path
import pickle

import torch
from torch.utils.data import Dataset


class OGDataset(Dataset):
    """S&Z dataset. Create pairs of S&Z samples in a dictionary format."""

    def __init__(self, data_path):
        """Initializes the list contains the dataset"""
        if torch.cuda.is_available():
            self.device = "cuda:0"
        else:
            self.device = "cpu"
        self.data_lst = [path for path in Path(data_path).glob('**/*.pkl') if "pre_structure" in str(path)]

    def __len__(self):
        """
    Returns:
      The number of samples the dataset contains
    """
        return len(self.data_lst)

    def __getitem__(self, idx):
        """ Allows to reach the sample in a certain index.

    Args:
      idx (int): The index of the sample we want to reach

    Returns:
      sample (tuple) - a tuple containing two elements:
      's' - a torch.Tensor who's the MSA Representation of the
      protein sequence, has shape `(s,r,c)`.
      'z' - a torch.Tensor who's the Pair Representation of the
      protein sequence, has shape `(r,r,c)`.
      labels - a torch tensor of size (s,) containing numbers from 0 to 19 (including)
      indicating the amino acid type in each position in the sequence.
    """
        with open(str(self.data_lst[idx]), 'rb') as f:
            out = pickle.load(f)

        return (out['s'].to(self.device), out['z'].to(self.device)), out['aatype'].to(self.device), self.data_lst[idx].parents[0].stem \
               + '__seed_' + self.data_lst[idx].stem.split('_')[-1]

    def __append__(self, input):
        """ Allows to append a given batch of samples to the dataset.

    Args:
      input (tuple(torch.Tensor, torch.Tensor)): The sample we want to append.
      The first torch.Tensor is of the shape `(b,s,r,c)`
      The second torch.Tensor is of the shape `(b,r,r,c)`
    """
        x, y = input
        for i in range(x.size(0)):
            self.data_lst.append({"s": x[i, ...], "z": y[i, ...]})
