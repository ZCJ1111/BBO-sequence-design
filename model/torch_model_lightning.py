import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
import os
from utils.seq_utils import sequences_to_tensor



class CustomEarlyStopping(EarlyStopping):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.best_train_loss = float('inf')
        
    def on_train_epoch_end(self, trainer, pl_module):
        current_loss = trainer.callback_metrics['train_loss']
        if current_loss < self.best_train_loss:
            self.best_train_loss = current_loss
        super().on_train_epoch_end(trainer, pl_module)

class TorchModel(pl.LightningModule):
    def __init__(self, args, alphabet, net, **kwargs):
        super().__init__()
        self.args = args
        self.alphabet = alphabet
        self.net = net.to('cpu')
        self.loss_func = torch.nn.MSELoss()
        
    
    def forward(self, x):
        return self.net(x)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr)
        return optimizer
    
    def training_step(self, batch, batch_idx):
        feats, labels = batch
        preds = torch.squeeze(self(feats.to(self.device)), dim=-1)
        loss = self.loss_func(preds, labels)
        self.log('train_loss', loss)
        return loss
    
    # def validation_step(self, batch, batch_idx):
    #     feats, labels = batch
    #     preds = torch.squeeze(self(feats.to(self.device)), dim=-1)
    #     loss = self.loss_func(preds, labels)
    #     self.log('val_loss', loss)
    #     return loss
    
    # def test_step(self, batch, batch_idx):
    #     feats, labels = batch
    #     preds = torch.squeeze(self(feats.to(self.device)), dim=-1)
    #     loss = self.loss_func(preds, labels)
    #     self.log('test_loss', loss)
    #     return loss
    
   
    def train_dataloader(self, sequences, labels):
        one_hots = sequences_to_tensor(sequences, self.alphabet)
        labels = torch.from_numpy(labels).float()
        dataset_train = TensorDataset(one_hots, labels)
        loader_train = DataLoader(
            dataset=dataset_train, batch_size=self.args.batch_size, shuffle=True
        )
        return loader_train
    
    # def val_dataloader(self, sequences, labels):
    #     one_hots = sequences_to_tensor(sequences, self.alphabet)
    #     labels = torch.from_numpy(labels).float()
    #     dataset_val = TensorDataset(one_hots, labels)
    #     loader_val = DataLoader(
    #         dataset=dataset_val, batch_size=self.args.batch_size, shuffle=True
    #     )
    #     return loader_val
    
    # def test_dataloader(self, sequences, labels):
    #     one_hots = sequences_to_tensor(sequences, self.alphabet)
    #     labels = torch.from_numpy(labels).float()
    #     dataset_test = TensorDataset(one_hots, labels)
    #     loader_test = DataLoader(
    #         dataset=dataset_test, batch_size=self.args.batch_size, shuffle=True
    #     )
    #     return loader_tes
    
    
    def running(self, sequences, labels):
        # model_name='cnn'
        # save_path = os.path.join('./check_point/', model_name)
        # os.makedirs(save_path, exist_ok=True)

        trainer = pl.Trainer(
            gpus=1 if str(self.device) == "cuda:0" else 0,
            max_epochs = self.args.max_epochs,
            callbacks=[CustomEarlyStopping(
                monitor='train_loss',
                patience=self.args.patience,
                verbose=False,
                mode='min',
                check_on_train_epoch_end = True
        )]
        )
        pl.seed_everything(42)
        train_loader = self.train_dataloader(sequences, labels)
        trainer.logger._log_graph = True 
        trainer.fit(self, train_loader)
        
        best_loss = trainer.callbacks[0].best_train_loss
        return best_loss
    # def train(self, sequences, labels):
    #     # Input: - sequences: [dataset_size, sequence_length]
    #     #        - labels:    [dataset_size]

    #     self.net.train()
    #     loader_train = self.get_data_loader(sequences, labels)
    #     best_loss, num_no_improvement = np.inf, 0
    #     epoch_num = 1
    #     while (num_no_improvement < self.args.patience) and (epoch_num < self.args.max_epochs):
    #         loss_List = []
    #         for data in tqdm(loader_train, desc=f"epoch{epoch_num}"):
    #             loss = self.compute_loss(data)
    #             loss_List.append(loss.item())
    #             self.optimizer.zero_grad()
    #             loss.backward()
    #             self.optimizer.step()
    #         current_loss = np.mean(loss_List)
    #         if current_loss < best_loss:
    #             best_loss = current_loss
    #             num_no_improvement = 0
    #         else:
    #             num_no_improvement += 1
    #         epoch_num += 1

    #     return best_loss

    def get_fitness(self, sequences):
        # Input:  - sequences:   [batch_size, sequence_length]
        # Output: - predictions: [batch_size]

        one_hots = sequences_to_tensor(sequences, self.alphabet).to('cpu')
        self.eval()
        predictions = self.forward(one_hots)
        predictions = np.squeeze(predictions, axis=-1)
        predictions = predictions.detach().numpy()
        return predictions
    
