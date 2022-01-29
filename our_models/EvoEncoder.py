from openfold.model.evoformer import EvoformerStack, EvoformerBlock
from openfold.config import model_config

import torch
from torch import nn

class EvoEncoder(nn.Module):

    def __init__(self, num_blocks=1, train=False):

        super(EvoEncoder, self).__init__()
        self.aa_types = 20
        self.feature_dim = 384
        self.config = model_config("model_1", train=train)
        self.config["model"]["evoformer_stack"]["c_m"] = self.feature_dim
        self.config["model"]["evoformer_stack"]["no_blocks"] = num_blocks
        self.evoformer = EvoformerStack(**self.config["model"]["evoformer_stack"])
        self.l1 = nn.Linear(self.feature_dim, self.feature_dim)
        self.l2 = nn.Linear(self.feature_dim, self.aa_types)

    def forward(self, x):
        s = x[0]
        z = x[1][0]
        msa_mask = torch.ones_like(s, dtype=torch.float32)
        seq_mask = torch.ones(s.size()[0], dtype=torch.float32)
        pair_mask = seq_mask[..., None] * seq_mask[..., None, :]
        _, _, s_out = self.evoformer(
            s,
            z,
            msa_mask=None,
            pair_mask=None,
            chunk_size=self.config.globals.chunk_size,
            _mask_trans=self.config.model._mask_trans)

        y = s_out + torch.relu(self.l1(s_out))
        y = torch.softmax(self.l2(y), dim=1)
        return y
