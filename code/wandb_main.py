import argparse

import pandas as pd
from tqdm.auto import tqdm

import transformers
import torch
import torchmetrics
import pytorch_lightning as pl

import wandb
import yaml
import os
from datetime import datetime
from functools import partial

from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from models.baseline import Model
from dataloader import Dataset, Dataloader
from arguments import get_args


def main(args):
    with wandb.init() as run:
        cfg = wandb.config
        model_artifact = wandb.Artifact(
            args.model_name.replace('/','_'),
            type='model',
            metadata=dict(cfg)
        )

        # dataloader와 model을 생성합니다.
        dataloader = Dataloader(args.model_name,
                                cfg.batch_size,
                                args.shuffle,
                                args.train_path,
                                args.dev_path,
                                args.test_path,
                                args.predict_path,
                                )

        model = Model(
            args.model_name,
            cfg.lr,
            args.weight_decay,
            args.loss
        )

        # Logger 생성
        wandb_logger = WandbLogger(log_model="all")

        #EarlyStopping
        earlystopping = EarlyStopping(monitor='val_pearson', patience=args.patience, mode='max')

        # Callback 생성
        checkpoint_callback = ModelCheckpoint(
            save_top_k=4,
            monitor="val_pearson",
            mode="max",
        )

        # learning rate monitor
        lr_monitor = LearningRateMonitor(logging_interval='step')

        # gpu가 없으면 accelerator='cpu', 있으면 accelerator='gpu'
        trainer = pl.Trainer(
            accelerator='gpu',
            logger=wandb_logger,
            callbacks=[checkpoint_callback, earlystopping, lr_monitor],
            max_epochs=cfg.max_epoch,
            log_every_n_steps=100
        )

        # Train part
        trainer.fit(model=model, datamodule=dataloader)

        MODEL_FILE = "{}_{}.pt".format(args.model_name.replace('/','_'), datetime.now().strftime('%Y%m%d_%H%M%S'))
        MODEL_PATH = os.path.join(args.ckpt_path, MODEL_FILE)
        
        trainer.save_checkpoint(MODEL_PATH)
        model_artifact.add_file(MODEL_PATH)
        wandb.save(MODEL_PATH)
        run.log_artifact(model_artifact)

        trainer.test(model=model, datamodule=dataloader)



if __name__ == '__main__':
    args = get_args()
    main_with_args = partial(main, args)

    # Define sweep config
    # https://docs.wandb.ai/guides/sweeps/define-sweep-configuration
    # https://github.com/borisdayma/lightning-kitti/blob/master/sweep.yaml
    sweep_configuration = {
        'method': 'random', # [grid, random, bayes] 중 택1
        'name': 'sweep_test',
        'metric': {'goal': 'minimize', 'name': 'val_loss'}, # 어떤 metric을 최적화하는게 목적인가?
        'parameters': 
        {
            'batch_size': {'distribution':'categorical',
                           'values': [8,16]},
            'lr': {'value' : args.learning_rate},
            'max_epoch':{'distribution':'constant',
                         'value':args.max_epoch}
        }
    }

    # Initialize sweep by passing in config. (Optional) Provide a name of the project.
    sweep_id = wandb.sweep(sweep=sweep_configuration)

    wandb.agent(sweep_id,
                function=main_with_args,
                count=args.run_count)
