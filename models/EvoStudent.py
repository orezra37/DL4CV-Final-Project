from openfold.model.embedders import InputEmbedder, ExtraMSAEmbedder
from openfold.model.evoformer import EvoformerStack, EvoformerBlock, ExtraMSAStack

from openfold.config import model_config

from openfold.utils.feats import build_extra_msa_feat
import torch
from torch import nn

class EvoStudent(nn.Module):

    def __init__(self, num_blocks=1, train=False, split_gpus=False):

        super(EvoStudent, self).__init__()

        # constants
        self.aa_types = 20

        # configurations
        self.split_gpus = split_gpus
        self.config = model_config("model_3", train=train)
        self.config["model"]["evoformer_stack"]["no_blocks"] = num_blocks
        # self.config["model"]["evoformer_stack"]["c_m"] = 384
        # self.config["model"]["evoformer_stack"]["inf"] = 1e4

        # architecture
        self.input_embedder = InputEmbedder(**self.config["model"]["input_embedder"])
        # self.extra_msa_embedder = ExtraMSAEmbedder(**self.config["model"]["extra_msa"]["extra_msa_embedder"])
        # self.extra_msa_stack = ExtraMSAStack(**self.config["model"]["extra_msa"]["extra_msa_stack"])
        self.evo1 = EvoformerStack(**self.config["model"]["evoformer_stack"])
        self.evo2 = EvoformerStack(**self.config["model"]["evoformer_stack"])
        self.evo3 = EvoformerStack(**self.config["model"]["evoformer_stack"])
        self.evo4 = EvoformerStack(**self.config["model"]["evoformer_stack"])
        self.layer_norm_m1 = nn.LayerNorm(self.config["model"]["evoformer_stack"]["c_m"])
        self.layer_norm_m2 = nn.LayerNorm(self.config["model"]["evoformer_stack"]["c_m"])
        self.layer_norm_m3 = nn.LayerNorm(self.config["model"]["evoformer_stack"]["c_m"])
        self.layer_norm_z1 = nn.LayerNorm(self.config["model"]["evoformer_stack"]["c_z"])
        self.layer_norm_z2 = nn.LayerNorm(self.config["model"]["evoformer_stack"]["c_z"])
        self.layer_norm_z3 = nn.LayerNorm(self.config["model"]["evoformer_stack"]["c_z"])

        self.input_embedder.cuda(0)
        if self.split_gpus:
            self.evo1.cuda(0)
            self.layer_norm_m1.cuda(1)
            self.layer_norm_z1.cuda(1)
            self.evo2.cuda(1)
            self.layer_norm_m2.cuda(2)
            self.layer_norm_z2.cuda(2)
            self.evo3.cuda(2)
            self.layer_norm_m3.cuda(3)
            self.layer_norm_z3.cuda(3)
            self.evo4.cuda(3)
        self.evos = [self.evo1, self.evo2, self.evo3, self.evo4]
        self.layer_norm_ms = [self.layer_norm_m1, self.layer_norm_m2, self.layer_norm_m3]
        self.layer_norm_zs = [self.layer_norm_z1, self.layer_norm_z2, self.layer_norm_z3]

    def forward(self, feats, GT_s=None):

        # Grab some data about the input
        batch_dims = feats["target_feat"].shape[:-2]
        no_batch_dims = len(batch_dims)
        n = feats["target_feat"].shape[-2]
        n_seq = feats["msa_feat"].shape[-3]
        device = feats["target_feat"].device

        s_lst = []
        m_lst = []
        z_lst = []

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
        # a = self.extra_msa_embedder(build_extra_msa_feat(feats))
        #
        # # # [*, N, N, C_z]
        # z = self.extra_msa_stack(
        #     a,
        #     z,
        #     msa_mask=feats["extra_msa_mask"],
        #     chunk_size=self.config.globals.chunk_size,
        #     pair_mask=pair_mask,
        #     _mask_trans=self.config.model._mask_trans,
        # )

        # Run MSA + pair embeddings through the trunk of the network
        # m: [*, S, N, C_m]
        # z: [*, N, N, C_z]
        # s: [*, N, C_s]
        # if GT_s is not None:
        #     m = GT_s.unsqueeze(0)
        for i in range(len(self.evos)):
            m, z, s = self.evos[i](
                m,
                z,
                msa_mask=msa_mask,
                pair_mask=pair_mask,
                chunk_size=self.config.globals.chunk_size,
                _mask_trans=self.config.model._mask_trans,
            )
            s = s.cuda((i+1) % 4)
            m = m.cuda((i+1) % 4)
            z = z.cuda((i+1) % 4)
            msa_mask = msa_mask.cuda((i+1) % 4)
            pair_mask = pair_mask.cuda((i+1) % 4)

            if i < 3:
                # [*, S_c, N, C_m]
                m = m + self.layer_norm_ms[i](m)
                # [*, N, N, C_z]
                z = z + self.layer_norm_zs[i](z)

        m_lst.append(m)
        z_lst.append(z)
        s_lst.append(s)
        # if self.split_gpus:
        #     m = m.to('cuda:1')
        #     z = z.to('cuda:1')
        #     msa_mask = msa_mask.to('cuda:1')
        #     pair_mask = pair_mask.to('cuda:1')

        # m, z, s = self.evo2(
        #     m,
        #     z,
        #     msa_mask=msa_mask,
        #     pair_mask=pair_mask,
        #     chunk_size=self.config.globals.chunk_size,
        #     _mask_trans=self.config.model._mask_trans,
        # )

        return s_lst, m_lst, z_lst