

from utils.seq_utils import *
import pandas as pd
import os
import math
import torch


import gpytorch
from gpytorch.models import ExactGP
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.kernels import RBFKernel
from torch.utils.data import Dataset, DataLoader
import gpytorch

class ProteinDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class GPModel(ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = RBFKernel()

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

protein_alphabet = "ACDEFGHIKLMNPQRSTVWY" 

df = pd.read_csv('unify-length/1ADQ_A.csv')
cdr3, energy = df['CDR3'], df['Energy']

  
alphabet = protein_alphabet
one_hot_cdr3 = []
for sequence in cdr3:
    one_hot = sequence_to_one_hot(sequence, alphabet)
    one_hot_cdr3.append(one_hot)

energy = torch.tensor(energy.values).unsqueeze(1)
one_hot_cdr3 = torch.stack(one_hot_cdr3).to('cpu')
train_dataset = ProteinDataset(one_hot_cdr3, energy)
train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)

likelihood = GaussianLikelihood()
model=GPModel(train_x=one_hot_cdr3, train_y=energy, likelihood=likelihood)
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
# Train the model
model.train()
likelihood.train()
training_iter = 10

for i in range(training_iter):
    epoch_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(one_hot_cdr3)
        
        loss = -mll(output, energy)
        loss_mean = loss.mean()
        loss_mean.backward()
        
        optimizer.step()
        
        print('Training Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            i, batch_idx * len(data), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss_mean.item()))
            
    print('Epoch: {}\tAverage Loss: {:.6f}'.format(i, epoch_loss / len(train_loader)))


