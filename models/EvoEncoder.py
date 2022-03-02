from openfold.model.evoformer import EvoformerStack, EvoformerBlock
from openfold.config import model_config

import torch
from torch import nn

class EvoEncoder(nn.Module):

    def __init__(self, num_blocks=1, train=False, dropout=0):

        super(EvoEncoder, self).__init__()

        # constants
        self.aa_types = 20
        self.feature_dim = 384

        # configurations
        self.config = model_config("model_3", train=train)
        self.config["model"]["evoformer_stack"]["c_m"] = self.feature_dim
        self.config["model"]["evoformer_stack"]["no_blocks"] = num_blocks

        # architecture
        self.evoformer = EvoformerStack(**self.config["model"]["evoformer_stack"])
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, self.aa_types)
        )

    def forward(self, x):
        s = x[0].half()
        z = x[1][0].half()
        _, _, s_out = self.evoformer(
            s,
            z,
            msa_mask=None,
            pair_mask=None,
            chunk_size=self.config.globals.chunk_size,
            _mask_trans=self.config.model._mask_trans)
        y = self.mlp_head(s_out)
        return y
