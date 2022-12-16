import json
import os

import numpy as np
import torch
import torch.nn as nn
from sequence_models.structure import Attention1d

from . import register_model, torch_model


class Decoder(nn.Module):
    def __init__(self, input_dim=1280, hidden_dim=512):
        super().__init__()
        self.dense_1 = nn.Linear(input_dim, hidden_dim)
        self.dense_2 = nn.Linear(hidden_dim, hidden_dim)
        self.attention1d = Attention1d(in_dim=hidden_dim)
        self.dense_3 = nn.Linear(hidden_dim, hidden_dim)
        self.dense_4 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = torch.relu(self.dense_1(x))
        x = torch.relu(self.dense_2(x))
        x = self.attention1d(x)
        x = torch.relu(self.dense_3(x))
        x = self.dense_4(x)
        return x


@register_model("esm1b")
class ESM1b_Attention1d(torch_model.TorchModel):
    def __init__(self):
        super().__init__()
        esm_dir_path = "./landscape_params/esm1b_landscape/esm_params"
        torch.hub.set_dir(esm_dir_path)
        self.encoder, self.alphabet = torch.hub.load(
            "facebookresearch/esm:main", "esm1b_t33_650M_UR50S"
        )
        self.tokenizer = self.alphabet.get_batch_converter()
        self.decoder = Decoder()

    def forward(self, x):
        x = self.encoder(x, repr_layers=[33], return_contacts=False)["representations"][33]
        x = self.decoder(x)
        return x
