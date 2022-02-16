from pathlib import Path
import pickle

import torch
from torch.utils.data import Dataset


class StudentDataset(Dataset):
    """S&Z dataset. Creates pairs of S&Z samples in a dictionary format."""

    def __init__(self, data_path):
        """Initializes the list contains the dataset"""

        self.folders = list(Path(data_path).glob('*'))
        self.data_lst = []
        for folder in self.folders:
            i = 0
            pre_structure = Path(folder, f'pre_structure_{i}.pkl')
            batch = Path(folder, f'batch_{i}.pkl')
            while pre_structure.exists() and batch.exists():
                self.data_lst.append((batch, pre_structure))
                i += 1
                pre_structure = Path(folder, f'pre_structure_{i}.pkl')
                batch = Path(folder, f'batch_{i}.pkl')

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
          's' - a torch.Tensor whose the MSA Representation of the
          protein sequence, has shape `(s,r,c)`.
          'z' - a torch.Tensor whose the Pair Representation of the
          protein sequence, has shape `(r,r,c)`.
          labels - a torch tensor of size (s,) containing numbers from 0 to 19 (including)
          indicating the amino acid type in each position in the sequence.
        """
        with open(str(self.data_lst[idx][0]), 'rb') as f:
            batch = pickle.load(f)

        with open(str(self.data_lst[idx][1]), 'rb') as f:
            out = pickle.load(f)

        return batch, (out['s'].cuda(), out['z'].cuda())