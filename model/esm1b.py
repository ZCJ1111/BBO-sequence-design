import gpytorch
import numpy as np
import torch
import esm
import torch.nn as nn
from sequence_models.structure import Attention1d
from torch.utils.data import DataLoader, TensorDataset

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
        mean_x = self.mean_module(x)
        # print('mean x',mean_x)
        covar_x = self.covar_module(x)
        x = gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        return x


class ESM1bAttention1d(nn.Module):
    def __init__(self, args):
        super().__init__()
        # esm_dir_path = args.torch_hub_cache
        # torch.hub.set_dir(esm_dir_path)
        # self.encoder, self.alphabet = torch.hub.load(
        #     "facebookresearch/esm:main", "esm2_t33_650M_UR50D"
        # )
        self.encoder, self.alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.tokenizer = self.alphabet.get_batch_converter()
        self.decoder = Decoder()

    def forward(self, x):
        x = self.encoder(x, repr_layers=[33], return_contacts=False)["representations"][33]
        x = self.decoder(x)
        return x


@register_model("esm1b")
class ESM1bModel(torch_model.TorchModel):
    def __init__(self, args, alphabet, **kwargs):
        model = ESM1bAttention1d(args)
        super().__init__(args, alphabet=alphabet, net=model, tokenizer=model.tokenizer, **kwargs)

    def get_data_loader(self, sequences, labels):
        # Input:  - sequences:    [dataset_size, sequence_length]
        #         - labels:       [dataset_size]
        # Output: - loader_train: torch.utils.data.DataLoader

        data = [(i, seq) for i, seq in enumerate(sequences)]
        *_, batch_tokens = self.net.tokenizer(data)
        labels = torch.from_numpy(labels).float()
        dataset_train = TensorDataset(batch_tokens, labels)
        loader_train = DataLoader(
            dataset=dataset_train, batch_size=self.args.batch_size, shuffle=True, num_workers=8
        )
        return loader_train

    def get_fitness(self, sequences):
        # Input:  - sequences:   [batch_size, sequence_length]
        # Output: - predictions: [batch_size]

        self.net.eval()
        with torch.no_grad():
            data = [(i, seq) for i, seq in enumerate(sequences)]
            *_, batch_tokens = self.net.tokenizer(data)
            batch_tokens = batch_tokens.to(self.device)
            predictions = self.net(batch_tokens).cpu().detach().numpy()
        predictions = np.squeeze(predictions, axis=-1)
        return predictions
