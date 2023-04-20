import argparse

import pandas as pd

from tqdm.auto import tqdm

import transformers
import torch
import torchmetrics
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import nn
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='snunlp/KR-ELECTRA-discriminator', type=str)

    plm = transformers.AutoModelForSequenceClassification.from_pretrained(
                pretrained_model_name_or_path=args.model_name, num_labels=1)

    a = 0
    for name, param in plm.named_parameters():
        a += 1
        print(name)

    print(f'the number of parameters is {a}')
