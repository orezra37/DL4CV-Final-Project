import os
import sys
import time
import pickle
import random
import argparse
import yaml
import numpy as np
from pathlib import Path

# torch imports
import torch
from torch import autograd
from torch.nn import MSELoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from openfold.utils.tensor_utils import tensor_tree_map

# openfold imports
from openfold.np.residue_constants import ID_TO_HHBLITS_AA

# my imports
from models.backenfold import Backenfold


from datetime import date
import logging

logging.basicConfig(stream=sys.stdout)

# A hack to get OpenMM and PyTorch to peacefully coexist
os.environ["OPENMM_DEFAULT_PLATFORM"] = "CUDA"  # "OpenCL"


import torch
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
from openfold.config import model_config
from openfold.data import templates, feature_pipeline, data_pipeline
from openfold.np import residue_constants, protein
import openfold.np.relax.relax as relax
from openfold.utils.import_weights import (
    import_jax_weights_,
)
from openfold.utils.tensor_utils import (
    tensor_tree_map,
)
from openfold.utils.loss import AlphaFoldLoss

from scripts.utils import add_data_args

# data transformations
from openfold.data.data_transforms import make_seq_mask, make_atom14_masks




MAX_CAPACITY_LENGTH = 400
EPS = 1

dataset_id_to_aatype = {
    0: "A",
    1: "R",
    2: "N",
    3: "D",
    4: "C",
    5: "Q",
    6: "E",
    7: "G",
    8: "H",
    9: "I",
    10: "L",
    11: "K",
    12: "M",
    13: "F",
    14: "P",
    15: "S",
    16: "T",
    17: "W",
    18: "Y",
    19: "V",
    20: "-",
}


def train(args):

    """Train loop for OG-former based on EVO-former."""

    # torch.set_default_dtype(torch.float16)
    config = model_config(args.model_name, train=True)
    config.data.common.max_recycling_iters = args.num_recycles
    Path(f'./checkpoints/Backenfold/{args.name}').mkdir(exist_ok=True, parents=True)
    writer = SummaryWriter('logs/Backenfold/' + args.name)

    my_data_pipeline = data_pipeline.DataPipeline(None)
    my_feature_pipeline = feature_pipeline.FeaturePipeline(config.data)

    model = Backenfold(config)
    import_jax_weights_(model, f'/shareDB/alphafold/params/params_{args.model_name}.npz', version=args.model_name)
    model.cuda()
    criterion = AlphaFoldLoss(config.loss)

    data = my_data_pipeline.process_pdb(pdb_path=args.structure, alignment_dir=args.msa)
    structure_feats = my_feature_pipeline.process_features(data, 'eval')
    # Remove the recycling dimension
    # ref_feats = {}
    # for k, v in structure_feats.items():
    #     ref_feats[k] = v.clone()

    if args.sequence is None:
        logits = torch.rand_like(structure_feats["target_feat"])
        structure_feats['target_feat'] = logits - logits.mean(-2, keepdims=True)
    for k, v in structure_feats.items():
        structure_feats[k] = v.cuda()

    model.train()

    #optimizer setup
    structure_feats["target_feat"].requires_grad_(True)
    optimizer = Adam([structure_feats["target_feat"]], lr=1e-3)  # TODO: disregard first and last position, and set them as

    for _ in range(args.num_steps):
        optimizer.zero_grad()
        out = model(structure_feats)
        ref_feats = tensor_tree_map(lambda t: t[..., -1].cuda(), structure_feats)
        loss = criterion(out, ref_feats)
        print(loss.item(), '\t', compute_drmsd(out["final_atom_positions"], ref_feats["all_atom_positions"]))
        loss.backward()
        optimizer.step()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--debug', action='store_true')
    p.add_argument('--sequence', type=str, help='path to initial sequence, otherwise a random sequence is initialized')
    p.add_argument('--msa', type=str, help='precomputed sequence alignment path')
    p.add_argument('--structure', required=True, type=str, help='path to pdb file of desired structure')
    p.add_argument('--name', type=str, default='untitled', help='experiment name (for checkpoints and logs)')
    p.add_argument('--model_name', type=str, default='model_1', help='AF model name')
    p.add_argument('--num_recycles', type=int, default=0, help="number of recycling iterations")
    p.add_argument('--num_steps', type=int, default=300, help="number of optimization steps")
    args = p.parse_args()
    if args.debug:
        import pydevd_pycharm
        pydevd_pycharm.settrace('10.96.3.148', port=123, stdoutToServer=True, stderrToServer=True)

    train(args)
