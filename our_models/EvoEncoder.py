from openfold.model.evoformer import EvoformerStack, EvoformerBlock
from openfold.config import model_config

from torch import nn

class EvoEncoder(nn.Module):

    def __init__(self, train=False):
        super(EvoEncoder, self).__init__()
        self.config = model_config("model_1", train=train)
        self.evoformer = EvoformerStack(**self.config["evoformer_stack"])

    def forward(self, s, z):
        msa_mask = torch.ones_like(s, dtype=torch.float32)
        seq_mask = torch.ones_like(s.size()[0], dtype=torch.float32)
        pair_mask = seq_mask[..., None] * seq_mask[..., None, :]
        _, _, s_out = self.evoformer(
            s,
            z,
            msa_mask=msa_mask,
            pair_mask=pair_mask,
            chunk_size=self.config.globals.chunk_size,
            _mask_trans=self.config._mask_trans)
        return s_out
