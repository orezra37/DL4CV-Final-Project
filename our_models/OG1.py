import torch
import torch.nn as nn


class OG1(nn.Module):
""" Model of a naive encoder net """



    def __init__(self, num_heads):
        super(OG1, self).__init__()
        # Parameters
        self.num_heads = num_heads
        self.res = 384

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
        # Transformer Layer, keeps the option to change our individual transformer
        self.transformer = nn.Sequential(
            self.default_transformer
        )
        self.mlp = nn.Sequential(
            self.linear1,
            self.sigmoid,
            self.linear2
        )


    def forward(self, s):
        y = self.transformer(s)

        return y
