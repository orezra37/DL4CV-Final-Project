from pathlib import Path
import pickle

import torch
from torch.utils.data import Dataset


class OGDataset(Dataset):
  """S&Z dataset. Creates pairs of S&Z samples in a dictionary format."""
  
  def __init__(self, data_path):
    """Initializes the list contains the dataset"""
    self.data_lst = list(Path(data_path).glob('*.pkl'))

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
    with open(str(self.data_lst[idx]), 'rb') as f:
      out = pickle.load(f)

    return (out['s'].cuda(), out['z'].cuda()), out['aatype'].cuda()

  def __append__(self, input):
    """ Allows to append a given batch of samples to the dataset.

    Args:
      input (tuple(torch.Tensor, torch.Tensor)): The sample we want to append.
      The first torch.Tensor is of the shape `(b,s,r,c)`
      The second torch.Tensor is of the shape `(b,r,r,c)`
    """
    x, y = input
    for i in range(x.size(0)):
      self.data_lst.append({"s": x[i,...], "z": y[i,...]})
