import torch
import torch.nn as nn


class OG1(nn.Module):
    """ Model of a naive encoder net.
    The result has the same size as the input.
    """

    def __init__(self, num_heads):
        super(OG1, self).__init__()
        # Parameters
        self.num_heads = num_heads
        self.res = 384  # default size of residue

        # Layers
        self.default_transformer = nn.Transformer(nhead=self.num_heads,
                                                  num_encoder_layers=6,
                                                  dropout=0.1)
        self.linear1 = nn.Linear(in_features=self.res,
                                 out_features=self.res)
        self.linear2 = nn.Linear(in_features=self.res,
                                 out_features=self.res)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()
        self.loss = nn.CrossEntropyLoss()

        # Final Layers
        # Transformer Layer, keeps the option to change to our individual transformer
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
        y = torch.cat((s.unsqueeze(), z), dim=1)
        y = self.transformer(y)
        y = self.mlp(y)
        y = self.softmax(y)
        return y

    @staticmethod
    def cat_sz(self, s, z):
        """
        Args:
            s and z
        Returns:
            y: cat s with the relevant column in z. has shape of ()
        """
        torch.cat((z[r, c], s[r], s[c]))
        y = torch.cat((s, z), dim=1)
        return y
