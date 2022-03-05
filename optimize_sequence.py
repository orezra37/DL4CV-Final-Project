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
from openfold.utils.loss import *

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

def init_logits(seq_len):
    logits = 0.01 * torch.rand(seq_len, 20, 1)
    logits -= logits.mean(-1, keepdims=True)
    logits.requires_grad_(True)
    return logits

def get_soft_seq(logits):
    # add torch.noise to logits?
    seq_soft = torch.softmax(logits) # with temperature?
    return seq_soft

def get_hard_seq(seq_soft):
    seq_hard = torch.one_hot(seq_soft.argmax(-1), 20)
    with torch.no_grad():
        seq_hard = seq_hard - seq_soft
    seq_hard += seq_soft
    return seq_hard

def update_seq(seq, inputs):
    target_feat = torch.zeros_like(inputs["target_feat"])
    target_feat[:, 1:21, :] = seq
    inputs.update({'target_feat': target_feat})

def AFLoss_with_dict(config):

    def compute_losses(out, ref):
        if "violation" not in out.keys() and config.violation.weight:
            out["violation"] = find_structural_violations(
                ref,
                out["sm"]["positions"][-1],
                **config.violation,
            )

        if "renamed_atom14_gt_positions" not in out.keys():
            ref.update(
                compute_renamed_ground_truth(
                    ref,
                    out["sm"]["positions"][-1],
                )
            )

        loss_fns = {
            "distogram": lambda: distogram_loss(
                logits=out["distogram_logits"],
                **{**ref, **config.distogram},
            ),
            "experimentally_resolved": lambda: experimentally_resolved_loss(
                logits=out["experimentally_resolved_logits"],
                **{**ref, **config.experimentally_resolved},
            ),
            "fape": lambda: fape_loss(
                out,
                ref,
                config.fape,
            ),
            "lddt": lambda: lddt_loss(
                logits=out["lddt_logits"],
                all_atom_pred_pos=out["final_atom_positions"],
                **{**ref, **config.lddt},
            ),
            "masked_msa": lambda: masked_msa_loss(
                logits=out["masked_msa_logits"],
                **{**ref, **config.masked_msa},
            ),
            "supervised_chi": lambda: supervised_chi_loss(
                out["sm"]["angles"],
                out["sm"]["unnormalized_angles"],
                **{**ref, **config.supervised_chi},
            ),
            "violation": lambda: violation_loss(
                out["violation"],
                **ref,
            ),
            "tm": lambda: tm_loss(
                logits=out["tm_logits"],
                **{**ref, **out, **config.tm},
            ),
        }

        cum_loss = 0.
        loss_dict = {}
        for loss_name, loss_fn in loss_fns.items():
            weight = config[loss_name].weight
            if weight:
                loss = loss_fn()
                loss_dict[f'Loss/{loss_name}'] = loss.item()
                if (torch.isnan(loss) or torch.isinf(loss)):
                    logging.warning(f"{loss_name} loss is NaN. Skipping...")
                    loss = loss.new_tensor(0., requires_grad=True)
                cum_loss = cum_loss + weight * loss

        loss_dict['Loss/total loss'] = cum_loss.item()
        return loss_dict, cum_loss

    return compute_losses

def config_loss_weights(weight_dict, config):
    for loss_name, weight in weight_dict.items():
        config[loss_name] = weight

def log_losses(writer, loss_dict, step_i):
    for k, v in loss_dict.items():
        writer.add_scalar(k, v, step_i)

def log_metrics(writer, out, ref, step_i):
    
    metric_fns = {
        "plddt": lambda: torch.mean(out['plddt']).item(),
        "drmsd": lambda: compute_drmsd(out["final_atom_positions"], ref["all_atom_positions"]).mean().item()
    }

    for k, v in metric_fns.items():
        writer.add_scalar(f'Metric/{k}', v(), step_i)

def optimize(args):

    """optimize sequence using AF gradients"""

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
    criterion = AFLoss_with_dict(config.loss) # AlphaFoldLoss(config.loss)#

    data = my_data_pipeline.process_pdb(pdb_path=args.structure, alignment_dir=args.msa)
    structure_feats = my_feature_pipeline.process_features(data, 'eval')

    if args.sequence is None:
        logits = init_logits(structure_feats['seq_length'])

    for k, v in structure_feats.items():
        structure_feats[k] = v.cuda()

    model.train()

    #optimizer setup
    optimizer = Adam([logits], lr=1e-3)  # TODO: disregard first and last position, and set them as

    # optimization loop
    print('starting optimization')
    ref_feats = tensor_tree_map(lambda t: t[..., -1].clone().detach().cuda(), structure_feats)
    for step_i in range(args.num_steps):
        update_seq(logits, structure_feats)
        optimizer.zero_grad()
        out = model(structure_feats)
        loss_dict, loss = criterion(out, ref_feats)
        log_losses(writer, loss_dict, step_i)
        log_metrics(writer, out, ref_feats, step_i)
        # print(loss.item(), '\t', compute_drmsd(out["final_atom_positions"], ref_feats["all_atom_positions"]))
        loss.backward()
        optimizer.step()
        print(step_i)

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

    optimize(args)