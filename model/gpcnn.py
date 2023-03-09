import gpytorch
import torch
import torch.nn as nn

from . import register_model, torch_model


class GPCNN(ExactGP,GPyTorchModel):
    _num_outputs = 1
    def __init__(self, num_input_channels,train_X,train_Y, num_filters=32, hidden_dim=128, kernel_size=5):
        super().__init__(train_X, train_Y.squeeze(-1), GaussianLikelihood())
        self.conv_1 = nn.Conv1d(num_input_channels, num_filters, kernel_size, padding="valid")
        self.conv_2 = nn.Conv1d(num_filters, num_filters, kernel_size, padding="same")
        self.conv_3 = nn.Conv1d(num_filters, num_filters, kernel_size, padding="same")
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        self.dense_1 = nn.Linear(num_filters, hidden_dim)
        self.dense_2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout_1 = nn.Dropout(0.25)
        self.dense_3 = nn.Linear(hidden_dim, 1)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
    
    def forward(self, x):
        # Input:  [batch_size, num_input_channels, sequence_length]
        # Output: [batch_size, 1]

        x = torch.relu(self.conv_1(x))
        x = torch.relu(self.conv_2(x))
        x = torch.relu(self.conv_3(x))
        x = torch.squeeze(self.global_max_pool(x), dim=-1)
        x = torch.relu(self.dense_1(x))
        x = torch.relu(self.dense_2(x))
        x = self.dropout_1(x)
        x = self.dense_3(x)
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


@register_model("gpcnn")
class ConvolutionalNetworkModel(torch_model.TorchModel):
    def __init__(self, args, alphabet, **kwargs):
        super().__init__(args, alphabet, net=CNN(num_input_channels=len(alphabet)))
    self.loss_func = gpytorch.mlls.ExactMarginalLogLikelihood(self.net.likelihood, self.net)

    
    def get_data_loader(self, sequences, labels):
        one_hots = sequences_to_tensor(sequences, self.alphabet).float()
        labels = torch.from_numpy(labels).float()
        dataset_train = TensorDataset(one_hots, labels)
        loader_train = DataLoader(
            dataset=dataset_train, batch_size=self.args.batch_size, shuffle=True
        )
        return loader_train
    
    def compute_loss(self,data):
        feats, labels = data
        outputs =torch.squeeze(self.net(feats.to(self.device))dim=-1)
        loss = self.loss_func(outputs, labels.to(self.device))

        return loss
    
    def train(self, sequences, labels):
        # Input: - sequences: [dataset_size, sequence_length]
        #        - labels:    [dataset_size]

        self.net.train()
        self.net.likelihood.train()
        loader_train = self.get_data_loader(sequences, labels)
        best_loss, num_no_improvement = np.inf, 0
        epoch_num = 1
        while (num_no_improvement < self.args.patience) and (epoch_num < self.args.max_epochs):
            loss_List = []
            for data in tqdm(loader_train, desc=f"epoch{epoch_num}"):
                loss = self.compute_loss(data)
                loss_List.append(loss.item())
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            current_loss = np.mean(loss_List)
            if current_loss < best_loss:
                best_loss = current_loss
                num_no_improvement = 0
            else:
                num_no_improvement += 1
            epoch_num += 1

        return best_loss

    def get_fitness(self, sequences):
        # Input:  - sequences:   [batch_size, sequence_length]
        # Output: - predictions: [batch_size]

        self.net.eval()
        self.net.likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            mutation_sets, mutation_sets_mask = sequences_to_mutation_sets(
                sequences, self.alphabet, self.wt_sequence, self.context_radius
            )
            predictions = (
                self.net(mutation_sets.to(self.device), mutation_sets_mask.to(self.device)))
            
            observation = self.net.likelihood(predictions).sample()
        
        # predictions.mean
        # predictions.variance
        observation = observation.detach().numpy()  
        # predictions = np.squeeze(predictions, axis=-1)
        
        return observation



