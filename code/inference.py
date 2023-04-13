import argparse

import pandas as pd

from tqdm.auto import tqdm

import transformers
import torch
import torchmetrics
import pytorch_lightning as pl

from models.baseline import Model
from dataloader import Dataloader
from arguments import get_args


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='snunlp/KR-ELECTRA-discriminator', type=str)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--max_epoch', default=1, type=int)
    parser.add_argument('--shuffle', default=True)
    parser.add_argument('--learning_rate', default=1e-5, type=float)
    parser.add_argument('--train_path', default='./data/train.csv')
    parser.add_argument('--dev_path', default='./data/dev.csv')
    parser.add_argument('--test_path', default='./data/dev.csv')
    parser.add_argument('--predict_path', default='./data/test.csv')
    parser.add_argument('--augmentation', default=True)
    args = parser.parse_args(args=[])
    # args = get_args()
    # print(get_args())
    # dataloader와 model을 생성합니다.
    dataloader = Dataloader(args.model_name, args.batch_size, args.shuffle, args.train_path, args.dev_path,
                            args.test_path, args.predict_path)

    # gpu가 없으면 accelerator='cpu', 있으면 accelerator='gpu'
    trainer = pl.Trainer(accelerator='gpu', max_epochs=args.max_epoch, log_every_n_steps=1)

    # Inference part
    # 저장된 모델로 예측을 진행합니다.
    model = torch.load('model.pt')
    predictions = trainer.predict(model=model, datamodule=dataloader)

    # 예측된 결과를 형식에 맞게 반올림하여 준비합니다.
    predictions = list(round(float(i), 1) for i in torch.cat(predictions))

    # output 형식을 불러와서 예측된 결과로 바꿔주고, output.csv로 출력합니다.
    output = pd.read_csv('./data/sample_submission.csv')
    output['target'] = predictions
    output.to_csv('output.csv', index=False)
