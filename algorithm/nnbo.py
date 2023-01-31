import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

class Model(nn.Module):
    """defining a model that will be used for acquisition function"""
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(1, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class NN_BO():
    def __init__(self, model,kernel,gp):
        
        self.model = model
        self.kernel = kernel
        self.gp = gp
    
    def acquisition(self, x):
        mean, std = self.gp.predict(x, return_std=True)
        mean = mean.flatten()
        std = std.flatten()
        z = (mean - np.min(mean)) / std
        x_augmented = np.concatenate([x, z[:, np.newaxis]], axis=1)
        ei = mean - np.min(mean) + std * (2 * norm.cdf(z) - 1)
        ei = ei.flatten()
        ei_nn = self.model.predict(x_augmented).flatten()
        return ei_nn
    
    def optimize(self, X, y, bounds, gp, model):
        
        x = np.array([np.random.uniform(bounds[0], bounds[1], 1) for i in range(100)])
        ei_nn = self.acquisition(x, gp, model)
        x_next = x[np.argmax(ei_nn)]
    
        return x_next


    def run(self, num_iters):
        self.X = []
        self.y = []
        for i in range(num_iters):
            if i == 0:
                x_next = np.random.uniform(-5, 5, 1)
            else:
                x_next = self.optimize(self.X, self.y, self.bounds, self.gp, self.model)
            y_next = self.f(x_next)
            self.X.append(x_next)
            self.y.append(y_next)
            self.X = np.array(self.X)
            self.y = np.array(self.y)
            self.gp.fit(self.X, self.y)
            self.model.fit(np.concatenate([self.X, (self.y - np.min(self.y))[:, np.newaxis] / self.gp.y_std_], axis=1), self.acquisition(self.X, self.gp, self.model), epochs=10, verbose=0)

