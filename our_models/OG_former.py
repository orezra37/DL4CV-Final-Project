import torch
import torch.nn as nn
import json


class OGDefaultTransformer(nn.Module):
    """ Model of a naive encoder net.
    The result has the same size as the input.
    This version of model does not use the information of z.
    """

    def __init__(self, num_heads, num_encoder_layers, num_decoder_layers, model_name):
        super(OGDefaultTransformer, self).__init__()
        # Parameters
        self.num_heads = num_heads
        self.res = 384  # default size of residue
        self.s_features_num = 128  # default number of features per sequence
        self.pre_process = self.only_s
        self.num_classes = 20
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.model_name = model_name

        # Layers
        self.default_transformer = nn.Transformer(d_model=self.res, nhead=self.num_heads,
                                                  num_encoder_layers=num_encoder_layers,
                                                  num_decoder_layers=num_decoder_layers)
        self.linear1 = nn.Linear(in_features=self.res, out_features=self.res)
        self.linear2 = nn.Linear(in_features=self.res, out_features=self.num_classes)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        self.s_norm = nn.LayerNorm(self.res)
        self.norm = nn.LayerNorm(self.res)

        self.mlp = nn.Sequential(
            self.linear1,
            self.sigmoid,
            self.linear2
        )

    def forward(self, s, z):
        """
        s - has size of (batch_size, s, 384)
        z - has size of (batch_size, s, s, 128)
        """
        y = self.pre_process(s, z)
        y = y[:, torch.randperm(s.size()[1]), :]
        y = self.s_norm(y)
        y = self.default_transformer(y, s)
        y = self.norm(y)
        y = self.mlp(y)  # result has size of (s,20)
        y = self.softmax(y)  # result probability of each amino-acid
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
        res = z.size(2)
        for i in range(seq_len):
            for j in range(seq_len):
                y = torch.cat((z[i, j], s[i], s[j]))
        return torch.flatten(y)

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


class OGOriginalTransformer(nn.Module):
    """ Model of a naive encoder net.
    The result has the same size as the input.
    This version of model does not use the information of z.
    """

    def __init__(self, num_heads, model_name):
        super(OGOriginalTransformer, self).__init__()
        # Parameters
        self.num_heads = num_heads
        self.res = 384  # default size of residue
        self.s_features_num = 128  # default number of features per sequence
        self.pre_process = self.only_s
        self.num_classes = 20
        self.model_name = model_name

        # Layers
        self.linear1 = nn.Linear(in_features=self.res, out_features=self.res)
        self.linear2 = nn.Linear(in_features=self.res, out_features=self.num_classes)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        self.norm = nn.LayerNorm(self.res)
        self.s_norm = nn.LayerNorm(self.res)
        self.q = nn.Linear(in_features=self.res, out_features=self.res)
        self.k = nn.Linear(in_features=self.res, out_features=self.res)
        self.v = nn.Linear(in_features=self.res, out_features=self.res)
        self.att = nn.MultiheadAttention(embed_dim=self.res, num_heads=self.num_heads)

        self.mlp = nn.Sequential(
            self.linear1,
            self.sigmoid,
            self.linear2
        )

    def forward(self, s, z):
        """
        s - has size of (batch_size, s, 384)
        z - has size of (batch_size, s, s, 128)
        """
        y = self.pre_process(s, z)
        y = self.s_norm(y)
        q, k, v = self.q(y), self.k(y), self.v(y)
        y = self.att(q, k, v)[0]
        y = self.norm(y)
        y = self.mlp(y)  # result has size of (s,20)
        y = self.softmax(y)  # result probability of each amino-acid
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
        res = z.size(2)
        for i in range(seq_len):
            for j in range(seq_len):
                y = torch.cat((z[i, j], s[i], s[j]))
        return torch.flatten(y)

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


class ReverseOriginalOG(nn.Module):
    """ Model of a naive encoder net.
    The result has the same size as the input.
    This version of model does not use the information of z.
    """

    def __init__(self, config_path):

        super(ReverseOriginalOG, self).__init__()
        # Parameters
        self.conf = json.load(open(config_path))
        self.num_heads = self.conf['num_heads']
        self.model_name = self.conf['model_save_name']
        self.res = 384  # default size of residue
        self.s_features_num = 128  # default number of features per sequence
        self.num_classes = 20
        self.device = None

        # Layers
        self.linear0 = nn.Linear(in_features=self.num_classes, out_features=self.res)
        self.linear1 = nn.Linear(in_features=self.res, out_features=self.res)
        self.linear2 = nn.Linear(in_features=self.res, out_features=self.res)
        self.sigmoid = nn.Sigmoid()
        self.norm0 = nn.LayerNorm(self.res)
        self.norm1 = nn.LayerNorm(self.res)
        self.q = nn.Linear(in_features=self.res, out_features=self.res)
        self.k = nn.Linear(in_features=self.res, out_features=self.res)
        self.v = nn.Linear(in_features=self.res, out_features=self.res)
        self.att = nn.MultiheadAttention(embed_dim=self.res, num_heads=self.num_heads)
        self.relu = nn.ReLU()

        self.mlp = nn.Sequential(
            self.linear1,
            self.sigmoid,
            self.linear2
        )

    def forward(self, seq):
        """
        seq - One dimensional vector which represents the amino acid sequence
        :return
        s - latent space tensor which has shape (s, 384)
        """
        y = seq
        y = torch.nn.functional.one_hot(y, self.num_classes).type(torch.FloatTensor)  # has shape (seq, 20)
        y = self.linear0(y)  # has shape (seq, 384)
        y = self.relu(y)
        y = self.norm0(y)
        q, k, v = self.q(y), self.k(y), self.v(y)
        y = self.att(q, k, v)[0]
        y = self.norm1(y)
        y = self.mlp(y)  # result has size of (s,384)
        return y[0]
