import argparse

import pandas as pd

from tqdm.auto import tqdm
from data_augmentation.eda import EDA 
import transformers
import torch
import torchmetrics
import pytorch_lightning as pl


class Dataset(torch.utils.data.Dataset):
    def __init__(self, inputs, targets=[]):
        self.inputs = inputs
        self.targets = targets

    # 학습 및 추론 과정에서 데이터를 1개씩 꺼내오는 곳
    def __getitem__(self, idx):
        # 정답이 있다면 else문을, 없다면 if문을 수행합니다
        if len(self.targets) == 0:
            return torch.tensor(self.inputs[idx])
        else:
            return torch.tensor(self.inputs[idx]), torch.tensor(self.targets[idx])

    # 입력하는 개수만큼 데이터를 사용합니다
    def __len__(self):
        return len(self.inputs)


class Dataloader(pl.LightningDataModule):
    def __init__(self, model_name, batch_size, shuffle, train_path, dev_path, test_path, predict_path, augmentation):
        super().__init__()
        self.model_name = model_name
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.train_path = train_path
        self.dev_path = dev_path
        self.test_path = test_path
        self.predict_path = predict_path

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.predict_dataset = None

        self.augmentation = augmentation

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, max_length=160)
        self.target_columns = ['label']
        self.delete_columns = ['id']
        self.text_columns = ['sentence_1', 'sentence_2']
        
        
    def tokenizing(self, dataframe):
        data = []
        for idx, item in tqdm(dataframe.iterrows(), desc='tokenizing', total=len(dataframe)):
            # 두 입력 문장을 [SEP] 토큰으로 이어붙여서 전처리합니다.
            text = '[SEP]'.join([item[text_column] for text_column in self.text_columns])
            outputs = self.tokenizer(text, add_special_tokens=True, padding='max_length', truncation=True)
            data.append(outputs['input_ids'])

        return data

    def preprocessing(self, data, stage):
        # 안쓰는 컬럼을 삭제합니다.
        data = data.drop(columns=self.delete_columns)

        # augmentations
        if self.augmentation and stage == 'fit':
            aug_sent1 = []
            aug_sent2 = []
            aug_source = []
            aug_label = []
            aug_binary_label = []

            for idx, item in tqdm(data.iterrows(), desc='augmentation', total=len(data)):
                #print(item)
                
                sent1, sent2 = tuple([item[text_column] for text_column in self.text_columns])
                augmented_sentences1, augmented_sentences2 = EDA(sent1, num_aug=3), EDA(sent2, num_aug=3)
                for aug_sentence1, aug_sentence2 in zip(augmented_sentences1, augmented_sentences2):
                    aug_sent1.append(aug_sentence1)
                    aug_sent2.append(aug_sentence2)
                aug_source += [item['source']] * len(augmented_sentences1) 
                aug_label += [item['label']] * len(augmented_sentences1) 
                aug_binary_label += [item['binary-label']] * len(augmented_sentences1)

            augmented_data = pd.DataFrame({'source':aug_source, 'sentence_1':aug_sent1, \
                                        'sentence_2':aug_sent2, 'label':aug_label, \
                                        'binary-label':aug_binary_label}, index=list(range(len(aug_sent1))))
            data = pd.concat([data, augmented_data]).sample(frac=1).reset_index(drop=True)
        
        # 텍스트 데이터를 전처리합니다.
        inputs = self.tokenizing(data)

        # 타겟 데이터가 없으면 빈 배열을 리턴합니다.
        try:
            targets = data[self.target_columns].values.tolist()
        except:
            targets = []

        return inputs, targets

    def setup(self, stage='fit'):
        if stage == 'fit':
            # 학습 데이터와 검증 데이터셋을 호출합니다
            train_data = pd.read_csv(self.train_path)
            val_data = pd.read_csv(self.dev_path)
            train_data = pd.concat([train_data, val_data])
            
            num = len(train_data)
            #분리 작업(전처리 후에는 문장 위치가 사라지니 미리 나눔)
            train_data, val_data, test_data = train_data.iloc[0:int(num*0.7), :], train_data.iloc[int(num*0.7):int(num*0.85), :], train_data.iloc[int(num*0.85):, :]
            
            self.test_data = test_data
            
            #데이터 증강
            train_data2 = train_data.copy()
            train_data2.rename(columns={'sentence_1':'sentence_2', 'sentence_2':'sentence_1'}, inplace = True)
            train_data = pd.concat([train_data, train_data2])
            
            # 학습데이터 준비
            train_inputs, train_targets = self.preprocessing(train_data, stage)
            print('train_inputs:', len(train_inputs))
            print('train_targets:', len(train_targets))


            # 검증데이터 준비
            val_inputs, val_targets = self.preprocessing(val_data, stage)
            
            # train 데이터만 shuffle을 적용해줍니다, 필요하다면 val, test 데이터에도 shuffle을 적용할 수 있습니다
            self.train_dataset = Dataset(train_inputs, train_targets)
            self.val_dataset = Dataset(val_inputs, val_targets)
        else:
            # 평가데이터 준비
            #test_data = pd.read_csv(self.test_path)
            test_inputs, test_targets = self.preprocessing(self.test_data, stage)
            
            self.test_dataset = Dataset(test_inputs, test_targets)

            predict_data = pd.read_csv(self.predict_path)
            predict_inputs, predict_targets = self.preprocessing(predict_data, stage)
            self.predict_dataset = Dataset(predict_inputs, [])

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=args.shuffle)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size)

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(self.predict_dataset, batch_size=self.batch_size)


class Model(pl.LightningModule):
    def __init__(self, model_name, lr):
        super().__init__()
        self.save_hyperparameters()

        self.model_name = model_name
        self.lr = lr

        # 사용할 모델을 호출합니다.
        self.plm = transformers.AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=model_name, num_labels=1)
        # Loss 계산을 위해 사용될 L1Loss를 호출합니다.
        self.loss_func = torch.nn.L1Loss()

    def forward(self, x):
        x = self.plm(x)['logits']

        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_func(logits, y.float())
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_func(logits, y.float())
        self.log("val_loss", loss)

        self.log("val_pearson", torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze()))

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        self.log("test_pearson", torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze()))

    def predict_step(self, batch, batch_idx):
        x = batch
        logits = self(x)

        return logits.squeeze()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer


if __name__ == '__main__':
    # 하이퍼 파라미터 등 각종 설정값을 입력받습니다
    # 터미널 실행 예시 : python3 run.py --batch_size=64 ...
    # 실행 시 '--batch_size=64' 같은 인자를 입력하지 않으면 default 값이 기본으로 실행됩니다
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

    # dataloader와 model을 생성합니다.

    dataloader = Dataloader(args.model_name, args.batch_size, args.shuffle, args.train_path, args.dev_path,
                            args.test_path, args.predict_path, args.augmentation)

    model = Model(args.model_name, args.learning_rate)
    
    
    # gpu가 없으면 accelerator='cpu', 있으면 accelerator='gpu'
    trainer = pl.Trainer(accelerator='gpu', max_epochs=args.max_epoch, log_every_n_steps=1)
    
    # Train part
    trainer.fit(model=model, datamodule=dataloader)
    trainer.test(model=model, datamodule=dataloader)

    # 학습이 완료된 모델을 저장합니다.
    torch.save(model, 'model.pt')
