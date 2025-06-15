import torch
import numpy as np
import lightning as L
from model import *
from data import *
from pytorch_lightning.loggers import TensorBoardLogger

torch.set_float32_matmul_precision('medium')

def matirx_factorization():
    embedding_dims, max_epochs, batch_size = 30, 40, 512
    data = LitData(ML100kData(), batch_size=batch_size)
    model = LitMF(MF, num_factors=embedding_dims, num_users=data.num_users, num_items=data.num_items)
    logger = TensorBoardLogger("log", name=f"MF_{embedding_dims}")
    trainer = L.Trainer(max_epochs=max_epochs, accelerator="auto", logger=logger)
    trainer.fit(model, data)

def auto_rec():
    embedding_dims, max_epochs, batch_size = 30, 40, 256
    user_based = True
    data = LitAutoRecData(AutoRecData(user_based=user_based), batch_size=batch_size, num_workers=4)
    model = LitAutoRec(
        AutoRec, 
        num_hidden=embedding_dims, 
        num_users=data.num_users, 
        num_items=data.num_items,
        user_based=user_based
    )
    logger = TensorBoardLogger("log", name=f"AutoRec_{"user" if user_based else "item"}_{embedding_dims}")
    trainer = L.Trainer(max_epochs=max_epochs, accelerator="auto", logger=logger)
    trainer.fit(model, data)

def neural_mf():
    embedding_dims, max_epochs, batch_size = 30, 40, 256
    data = LitData(
        ML100KPairWise(test_sample_size=100),
        batch_size=batch_size,
        num_workers=4
    )

    model = LitNeuMF(
        num_users=data.num_users, num_items=data.num_items,
        embedding_dims=embedding_dims,
        hidden_dims=[10, 10, 10]
    )

    logger = TensorBoardLogger("log", name=f"NeuMF")
    trainer = L.Trainer(max_epochs=max_epochs, accelerator="auto", logger=logger)

    trainer.fit(model, data)

def caser():
    embedding_dims, max_epochs, batch_size = 30, 40, 512
    seq_len = 5
    data = LitData(
        ML100KSequence(seq_len=seq_len),
        batch_size=batch_size,
        num_workers=4
    )

    model = LitCaser(
        num_users=data.num_users, num_items=data.num_items,
        embedding_dims=embedding_dims,
        seq_len=seq_len
    )

    logger = TensorBoardLogger("log", name=f"caser")
    trainer = L.Trainer(max_epochs=max_epochs, accelerator="auto", logger=logger)

    trainer.fit(model, data)

def fm():
    embedding_dims, max_epochs, batch_size = 30, 40, 512
    data = LitData(
        CTRDataset(),
        batch_size=batch_size,
        num_workers=4
    )

    model = LitFM(
        feat_dims=data.dataset.feat_dims,
        embedding_dims=embedding_dims,
    )

    logger = TensorBoardLogger("log", name=f"fm")
    trainer = L.Trainer(max_epochs=max_epochs, accelerator="auto", logger=logger)

    trainer.fit(model, data)

def deepfm():
    embedding_dims, max_epochs, batch_size = 30, 40, 512
    data = LitData(
        CTRDataset(),
        batch_size=batch_size,
        num_workers=4
    )

    model = LitDeepFM(
        feat_dims=data.dataset.feat_dims,
        embedding_dims=embedding_dims,
    )

    logger = TensorBoardLogger("log", name=f"deepfm")
    trainer = L.Trainer(max_epochs=max_epochs, accelerator="auto", logger=logger)

    trainer.fit(model, data)

def test_loss():
    embedding_dims, max_epochs, batch_size = 30, 20, 512
    data = LitData(ML100kData(), batch_size=batch_size, num_workers=0)
    model = LitMF(MF, num_factors=embedding_dims, num_users=data.num_users, num_items=data.num_items)
    rating_matrix = np.zeros(
            (data.num_items, data.num_users), dtype=np.float32)
    print(rating_matrix[data.dataset.item_id, data.dataset.user_id].shape)
    print(data.dataset.item_id.shape, rating_matrix.shape)
    # rating_matrix[data.item_id]
    for batch in data.train_dataloader():
        x, y, z = batch
        print(x.dtype, y.dtype, z.dtype)
        print(x.shape, y.shape, z.shape)
        p = model.model(x, y)
        loss = model.get_loss(p, batch)
        print(loss)
        break

if __name__ == "__main__":
    # matirx_factorization()
    # auto_rec()
    # neural_mf()
    # caser()
    # fm()
    deepfm()
    # test_loss()