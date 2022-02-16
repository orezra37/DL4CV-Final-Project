from openfold.model.embedders import InputEmbedder, ExtraMSAEmbedder
from openfold.model.evoformer import EvoformerStack, EvoformerBlock, ExtraMSAStack

from openfold.config import model_config

from openfold.utils.feats import build_extra_msa_feat
import torch
from torch import nn

class EvoStudent(nn.Module):

    def __init__(self, num_blocks=1, train=False, dropout=0):

        super(EvoStudent, self).__init__()

        # constants
        self.aa_types = 20

        # configurations
        self.config = model_config("model_3", train=train)
        self.config["model"]["evoformer_stack"]["no_blocks"] = num_blocks
        # self.config["model"]["evoformer_stack"]["inf"] = 1e4

        # architecture
        self.input_embedder = InputEmbedder(**self.config["model"]["input_embedder"])
        self.extra_msa_embedder = ExtraMSAEmbedder(**self.config["model"]["extra_msa"]["extra_msa_embedder"])
        self.extra_msa_stack = ExtraMSAStack(**self.config["model"]["extra_msa"]["extra_msa_stack"])
        self.evoformer = EvoformerStack(**self.config["model"]["evoformer_stack"])

    def forward(self, feats):

        # Grab some data about the input
        batch_dims = feats["target_feat"].shape[:-2]
        no_batch_dims = len(batch_dims)
        n = feats["target_feat"].shape[-2]
        n_seq = feats["msa_feat"].shape[-3]
        device = feats["target_feat"].device

        # Prep some features
        seq_mask = feats["seq_mask"]
        pair_mask = seq_mask[..., None] * seq_mask[..., None, :]
        msa_mask = feats["msa_mask"]

        # Initialize the MSA and pair representations

        # m: [*, S_c, N, C_m]
        # z: [*, N, N, C_z]
        m, z = self.input_embedder(
            feats["target_feat"],
            feats["residue_index"],
            feats["msa_feat"],
        )
        # Embed extra MSA features + merge with pairwise embeddings
        # [*, S_e, N, C_e]
        a = self.extra_msa_embedder(build_extra_msa_feat(feats))

        # # [*, N, N, C_z]
        z = self.extra_msa_stack(
            a,
            z,
            msa_mask=feats["extra_msa_mask"],
            chunk_size=self.config.globals.chunk_size,
            pair_mask=pair_mask,
            _mask_trans=self.config.model._mask_trans,
        )

        # Run MSA + pair embeddings through the trunk of the network
        # m: [*, S, N, C_m]
        # z: [*, N, N, C_z]
        # s: [*, N, C_s]
        _, z, s = self.evoformer(
            m,
            z,
            msa_mask=msa_mask,
            pair_mask=pair_mask,
            chunk_size=self.config.globals.chunk_size,
            _mask_trans=self.config.model._mask_trans,
        )
        return s, z