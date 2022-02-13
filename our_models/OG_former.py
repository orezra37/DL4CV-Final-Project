import torch
import torch.nn as nn


class OG1(nn.Module):
    """ Model of a naive encoder net.
    The result has the same size as the input.
    This version of model does not use the information of z.
    """

    def __init__(self, num_heads):
        super(OG1, self).__init__()
        # Parameters
        self.num_heads = num_heads
        self.res = 384  # default size of residue
        self.s_features_num = 128  # default number of features per sequence
        self.pre_process_fun = self.only_s

        # Layers
        self.default_transformer = nn.Transformer(nhead=self.num_heads,
                                                  num_encoder_layers=6,
                                                  dropout=0)
        self.linear1 = nn.Linear(in_features=self.res,
                                 out_features=self.res)
        self.linear2 = nn.Linear(in_features=self.res,
                                 out_features=self.res)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()

        # Final Layers
        self.transformer = nn.Sequential(
            self.default_transformer
        )
        self.mlp = nn.Sequential(
            self.linear1,
            self.sigmoid,
            self.linear2
        )

    def forward(self, s, z):
        """
        s - has size of (s,384)
        z - has size of (s,s,128)
        """
        y = self.pre_process_fun(s, z)
        y = self.transformer(y)
        y = self.mlp(y)
        y = self.softmax(y)
        return y

    @staticmethod
    def cat_sz(s, z):
        """
        Args:
            s - has size of (s,384)
            z - has size of (s,s,128)
        Returns:
            y: cat s features with the match features in z. has shape of (s^2, 128+128+384)
        """
        seq_len = s.size(0)
        for i in range(seq_len):
            for j in range(seq_len):
                y = torch.cat((z[i,j], s[i], s[j]))
        return y.flatten()

    @staticmethod
    def only_s(s, z):
        """
        Args:
            s - has size of (s,384)
            z - has size of (s,s,128)
        Returns:
            only s
        """
        return s



