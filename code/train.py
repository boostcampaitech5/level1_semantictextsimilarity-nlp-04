import argparse

import pandas as pd

from tqdm.auto import tqdm

import transformers
import torch
import torchmetrics
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from models.baseline import Model
from dataloader import Dataloader
from arguments import get_args



if __name__ == '__main__':
    args = get_args()

    # dataloader와 model을 생성합니다.
    dataloader = Dataloader(
        args.model_name, 
        args.batch_size, 
        args.shuffle, 
        args.train_path, 
        args.dev_path,
        args.test_path, 
        args.predict_path
    )

    model = Model(
        args.model_name, 
        args.learning_rate, 
        args.weight_decay, 
        args.loss
    )

    # CSVLogger 생성
    logger = CSVLogger()

    # ModelCheckpoint 저장 callback
    checkpoint_callback = ModelCheckpoint(
        save_top_k=4,
        monitor="val_pearson",
        mode="max",
    )

    # EarlyStopping
    earlystopping = EarlyStopping(monitor='val_pearson', patience=args.patience, mode='max')

    # learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='step')


    # gpu가 없으면 accelerator='cpu', 있으면 accelerator='gpu'
    trainer = pl.Trainer(
        accelerator='gpu',
        logger=logger,
        callbacks=[checkpoint_callback, earlystopping, lr_monitor],
        max_epochs=args.max_epoch,
        log_every_n_steps=100,
    )
    
    
    # Train part
    trainer.fit(model=model, datamodule=dataloader)
    trainer.test(model=model, datamodule=dataloader)

    # 학습이 완료된 모델을 저장합니다.
    torch.save(model, 'model.pt')
