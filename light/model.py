"""
Model
"""

import torch
from torch import nn
import lightning as L
import torch.nn.functional as F
from typing import List
from torchmetrics import MeanSquaredError, RetrievalHitRate, Accuracy

from utils import bpr_loss

class MF(nn.Module):
    def __init__(self, num_factors:int, num_users:int, num_items:int, **kwargs):
        super().__init__(**kwargs)
        self.P = nn.Embedding(num_users, num_factors)
        self.Q = nn.Embedding(num_items, num_factors)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)

    def forward(self, user_id:torch.Tensor, item_id:torch.Tensor):
        P_u = self.P(user_id)
        Q_i = self.Q(item_id)
        b_u = self.user_bias(user_id).flatten()
        b_i = self.item_bias(item_id).flatten()
        outputs = (P_u * Q_i).sum(axis=1) + b_u + b_i
        return outputs

class LitMF(L.LightningModule):
    def __init__(self, model:nn.Module, lr:float=0.002, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.model = model(**kwargs)
        self.lr = lr
        self.rmse = MeanSquaredError()
        self.training_step_outputs = []
        self.validation_step_outputs = []

    def get_loss(self, pred_ratings:torch.Tensor, batch:tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
        return F.mse_loss(pred_ratings, batch[-1])

    def forward(self, batch:tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
        user_ids, item_ids, _ = batch
        return self.model(user_ids, item_ids)
        
    def training_step(self, batch:tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx:int):
        outputs = self(batch)
        loss = self.get_loss(outputs, batch)
        self.training_step_outputs.append(loss)
        return loss
        
    def validation_step(self, batch:tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx:int):
        outputs = self(batch)
        loss = self.get_loss(outputs, batch)
        self.validation_step_outputs.append(loss)
        self.update_metric(outputs, batch)
        return loss

    def update_metric(self, outputs:torch.Tensor, batch:tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
        _, _, gt = batch
        self.rmse.update(outputs, gt)
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), self.lr, weight_decay=1e-5)

    def on_train_epoch_end(self):
        epoch_average = torch.stack(self.training_step_outputs).mean()
        self.logger.experiment.add_scalar(
            "train/loss", epoch_average, self.current_epoch)
        self.training_step_outputs.clear()

    def on_validation_epoch_end(self):
        epoch_average = torch.stack(self.validation_step_outputs).mean()
        self.logger.experiment.add_scalar(
            "val/loss", epoch_average, self.current_epoch)
        self.logger.experiment.add_scalar(
            "val/mse", self.rmse.compute(), self.current_epoch)
        self.rmse.reset()
        self.validation_step_outputs.clear()

class AutoRec(nn.Module):
    def __init__(
        self, 
        num_hidden:int, 
        num_users:int, 
        num_items:int,
        user_based:bool=True,
        dropout:float=0.05
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.LazyLinear(num_hidden, bias=True), 
            nn.Sigmoid(),
            nn.Dropout(dropout)
        )
        
        if user_based:
            self.decoder = nn.LazyLinear(num_items, bias=True)
        else:
            self.decoder = nn.LazyLinear(num_users, bias=True)

    def forward(self, input: torch.Tensor):
        hidden = self.encoder(input)
        pred = self.decoder(hidden)
        return pred

class LitAutoRec(L.LightningModule):
    def __init__(self, model:nn.Module, lr:float=0.01, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.model = model(**kwargs)
        self.lr = lr
        self.rmse = MeanSquaredError()
        self.training_step_outputs = []
        self.validation_step_outputs = []

    def get_loss(self, pred_ratings:torch.Tensor, batch:torch.Tensor):
        mask = (batch > 0).to(torch.float32) # only consider observed ratings
        pred_ratings = pred_ratings * mask
        return F.mse_loss(pred_ratings, batch)

    def forward(self, batch:torch.Tensor):
        return self.model(batch)
        
    def training_step(self, batch:torch.Tensor, batch_idx:int):
        outputs = self(batch)
        loss = self.get_loss(outputs, batch)
        self.training_step_outputs.append(loss)
        return loss
        
    def validation_step(self, batch:torch.Tensor, batch_idx:int):
        outputs = self(batch)
        loss = self.get_loss(outputs, batch)
        self.validation_step_outputs.append(loss)
        self.update_metric(outputs, batch)
        return loss

    def update_metric(self, outputs:torch.Tensor, batch:torch.Tensor):
        mask = batch > 0
        self.rmse.update(outputs[mask], batch[mask])
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), self.lr, weight_decay=1e-5)

    def on_train_epoch_end(self):
        epoch_average = torch.stack(self.training_step_outputs).mean()
        self.logger.experiment.add_scalar(
            "train/loss", epoch_average, self.current_epoch)
        self.training_step_outputs.clear()

    def on_validation_epoch_end(self):
        epoch_average = torch.stack(self.validation_step_outputs).mean()
        self.logger.experiment.add_scalar(
            "val/loss", epoch_average, self.current_epoch)
        self.logger.experiment.add_scalar(
            "val/mse", self.rmse.compute(), self.current_epoch)
        self.rmse.reset()
        self.validation_step_outputs.clear()

class NeuMF(nn.Module):
    def __init__(self, embedding_dims: int, num_users: int, num_items: int, hidden_dims: List, **kwargs):
        super().__init__()
        self.P = nn.Embedding(num_users, embedding_dims)
        self.Q = nn.Embedding(num_items, embedding_dims)
        self.U = nn.Embedding(num_users, embedding_dims)
        self.V = nn.Embedding(num_items, embedding_dims)
        mlp = [nn.Linear(embedding_dims*2, hidden_dims[0]),
               nn.ReLU()]
        for i in range(len(hidden_dims) - 1):
            mlp += [nn.Linear(hidden_dims[i], hidden_dims[i+1]),
                    nn.ReLU()]
        self.mlp = nn.Sequential(*mlp)
        self.output_layer = nn.Linear(
            hidden_dims[-1] + embedding_dims, 1, bias=False)

    def forward(self, user_id, item_id) -> torch.Tensor:
        p_mf = self.P(user_id)
        q_mf = self.Q(item_id)
        gmf = p_mf * q_mf

        p_mlp = self.U(user_id)
        q_mlp = self.V(item_id)
        mlp = self.mlp(torch.cat([p_mlp, q_mlp], axis=-1))
        logit = self.output_layer(
            torch.cat([gmf, mlp], axis=-1))
        return logit

class LitNeuMF(L.LightningModule):
    def __init__(self, lr=0.002, hitrate_cutout=10, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.model = NeuMF(**kwargs)
        self.lr = lr
        self.hitrate = RetrievalHitRate(top_k=hitrate_cutout)
        self.training_step_outputs = []

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), self.lr, weight_decay=1e-5)

    def forward(self, user_id, item_id):
        return self.model(user_id, item_id)

    def training_step(self, batch, batch_idx):
        user_id, pos_item, neg_item = batch
        pos_score = self(user_id, pos_item)
        neg_score = self(user_id, neg_item)
        loss = bpr_loss(pos_score, neg_score)
        self.training_step_outputs.append(loss)
        return loss

    def validation_step(self, batch, batch_idx):
        user_id, item_id, is_pos = batch
        logit = self(user_id, item_id)
        score = torch.sigmoid(logit).reshape(-1,)
        self.hitrate.update(score, is_pos, user_id.to(torch.int64))
        return 

    def on_train_epoch_end(self):
        epoch_average = torch.stack(self.training_step_outputs).mean()
        self.logger.experiment.add_scalar(
            "train/loss", epoch_average, self.current_epoch)
        self.training_step_outputs.clear()

    def on_validation_epoch_end(self):
        self.logger.experiment.add_scalar(
            f"val/hit_rate@{self.hitrate.top_k}", self.hitrate.compute(), self.current_epoch)
        self.hitrate.reset()

class Caser(nn.Module):
    def __init__(self, embedding_dims, num_users, num_items,
                 L=5, num_hfilters=16, num_vfilters=4,
                 dropout=0.05, **kwargs):
        super().__init__()
        self.P = nn.Embedding(num_users, embedding_dims)
        self.Q = nn.Embedding(num_items, embedding_dims)

        self.num_hfilters = num_hfilters
        self.num_vfilters = num_vfilters
        # Vertical convolution
        self.conv_v = nn.Conv2d(1, num_vfilters, (L, 1))
        # Horizontal convolutions
        self.conv_h = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, num_hfilters, (h, embedding_dims)),
                nn.ReLU(),
                nn.AdaptiveMaxPool2d((1, 1)))
            for h in range(1, L+1)])
        # Fully-connected layer
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(
                num_vfilters*embedding_dims + num_hfilters*L,
                embedding_dims),
            nn.ReLU())
        self.Q_out = nn.Embedding(num_items, 2*embedding_dims)
        self.b_out = nn.Embedding(num_items, 1)

    def forward(self, user_id, seq, item_id):
        item_emb = self.Q(seq).unsqueeze(1)
        user_emb = self.P(user_id)

        v = self.conv_v(item_emb)
        h = torch.cat([filt(item_emb) for filt in self.conv_h], axis=-2)
        x = self.fc(torch.cat([v.flatten(1), h.flatten(1)], -1))
        x = torch.cat([x, user_emb], -1)
        logit = (self.Q_out(item_id)*x).sum(-1) + self.b_out(item_id).squeeze()
        return logit

class LitCaser(L.LightningModule):
    def __init__(self, lr=0.002, hitrate_cutout=10, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.model = Caser(**kwargs)
        self.lr = lr
        self.hitrate = RetrievalHitRate(top_k=hitrate_cutout)
        self.training_step_outputs = []

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), self.lr, weight_decay=1e-5)

    def forward(self, user_id, seq, item_id):
        return self.model(user_id, seq, item_id)

    def training_step(self, batch, batch_idx):
        user_id, seq, pos_item, neg_item = batch
        pos_logit = self(user_id, seq, pos_item)
        neg_logit = self(user_id, seq, neg_item)
        loss = bpr_loss(pos_logit, neg_logit)
        self.training_step_outputs.append(loss)
        return loss

    def validation_step(self, batch, batch_idx):
        user_id, seq, item_id, is_pos = batch
        logit = self(user_id, seq, item_id)
        score = torch.sigmoid(logit).reshape(-1,)
        self.hitrate.update(score, is_pos, user_id)
        return

    def on_train_epoch_end(self):
        avg_loss = torch.stack(self.training_step_outputs).mean()
        self.logger.experiment.add_scalar(
            "train/loss", avg_loss, self.current_epoch)
        self.training_step_outputs.clear()

    def on_validation_epoch_end(self):
        self.logger.experiment.add_scalar(
            f"val/hit_rate@{self.hitrate.top_k}",
            self.hitrate.compute(),
            self.current_epoch)
        self.hitrate.reset()

class FactorizationMachine(nn.Module):
    def __init__(self, feat_dims, embedding_dims, **kwargs):
        super().__init__()
        num_inputs = int(sum(feat_dims))
        self.embedding = nn.Embedding(num_inputs, embedding_dims)
        self.proj = nn.Embedding(num_inputs, 1)
        self.fc = nn.Linear(1, 1)
        for param in self.parameters():
            try:
                nn.init.xavier_normal_(param)
            finally:
                continue

    def forward(self, x, return_logit=False):
        v = self.embedding(x)
        interaction = 1/2*(v.sum(1)**2 - (v**2).sum(1)).sum(-1, keepdims=True)
        proj = self.proj(x).sum(1)
        logit = self.fc(proj + interaction).flatten()
        if return_logit:
            return logit
        else:
            return torch.sigmoid(logit)


class LitFM(L.LightningModule):
    def __init__(self, lr=0.002, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.model = FactorizationMachine(**kwargs)
        self.lr = lr
        self.train_acc = Accuracy(task="binary")
        self.test_acc = Accuracy(task="binary")
        self.training_step_outputs = []
        self.validation_step_outputs = []

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), self.lr, weight_decay=1e-5)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        ypred = self(x)
        loss = F.binary_cross_entropy(ypred, y.to(torch.float32))
        self.train_acc.update(ypred, y)
        self.training_step_outputs.append(loss)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        ypred = self(x)
        loss = F.binary_cross_entropy(ypred, y.to(torch.float32))
        self.test_acc.update(ypred, y)
        self.validation_step_outputs.append(loss)
        return {"loss": loss}
        
    def on_train_epoch_end(self):
        avg_loss = torch.stack(self.training_step_outputs).mean()
        acc = self.train_acc.compute()
        self.logger.experiment.add_scalar(
            "train/loss", avg_loss, self.current_epoch)
        self.logger.experiment.add_scalar(
            "train/acc", acc, self.current_epoch)
        self.training_step_outputs.clear()

    def on_validation_epoch_end(self):
        avg_loss = torch.stack(self.validation_step_outputs).mean()
        acc = self.test_acc.compute()
        self.logger.experiment.add_scalar(
            "val/loss", avg_loss, self.current_epoch)
        self.logger.experiment.add_scalar(
            "val/acc", acc, self.current_epoch)
        self.validation_step_outputs.clear()

def mlp_layer(in_dim, out_dim, dropout=0.0):
    return [
        nn.Linear(in_dim, out_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
    ]

class DeepFM(nn.Module):
    def __init__(self, feat_dims, embedding_dims,
                 mlp_dims=[30, 20, 10], dropout=0.1):
        super().__init__()
        num_inputs = int(sum(feat_dims))
        self.embed_output_dim = len(feat_dims) * embedding_dims
        self.embedding = nn.Embedding(num_inputs, embedding_dims)
        self.proj = nn.Embedding(num_inputs, 1)
        self.fc = nn.Linear(1, 1)
        self.mlp = nn.Sequential(
            *mlp_layer(self.embed_output_dim, mlp_dims[0], dropout),
            *[layer for i in range(len(mlp_dims) - 1)
              for layer in mlp_layer(mlp_dims[i], mlp_dims[i+1], dropout)],
            nn.Linear(mlp_dims[-1], 1))
        self.init_param()

    def init_param(self):
        for param in self.parameters():
            try:
                nn.init.xavier_normal_(param)
            finally:
                continue

    def forward(self, x):
        v = self.embedding(x)
        # Factorization Machine
        fm_interaction = 1/2*(v.sum(1)**2 - (v**2).sum(1)
                              ).sum(-1, keepdims=True)
        fm_proj = self.proj(x).sum(1)
        fm_logit = self.fc(fm_proj + fm_interaction).flatten()
        # MLP
        mlp_logit = self.mlp(v.flatten(1)).flatten()
        logit = fm_logit + mlp_logit
        return torch.sigmoid(logit)


class LitDeepFM(LitFM):
    def __init__(self, lr=0.002, **kwargs):
        super(LitFM, self).__init__()
        self.save_hyperparameters()
        self.model = DeepFM(**kwargs)
        self.lr = lr
        self.train_acc = Accuracy(task="binary")
        self.test_acc = Accuracy(task="binary")
        self.training_step_outputs = []
        self.validation_step_outputs = []