import torch
import pytorch_lightning as pl
import pandas as pd
import transformers
from tqdm import tqdm


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
    def __init__(self,
                 model_name,
                 batch_size,
                 shuffle,
                 train_path,
                 dev_path, 
                 test_path, 
                 predict_path
    ):
        super().__init__()
        self.save_hyperparameters()
        
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

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, max_length=320)
        self.target_columns = ['label']
        self.delete_columns = ['id']
        self.text_columns = ['sentence_1', 'sentence_2']

    def tokenizing(self, dataframe:pd.DataFrame):
        data = []
        for idx, item in tqdm(dataframe.iterrows(), desc='tokenizing', total=len(dataframe)):
            # 두 입력 문장을 [SEP] 토큰으로 이어붙여서 전처리합니다.
            text = '[SEP]'.join([item[text_column] for text_column in self.text_columns])
            outputs = self.tokenizer(text, add_special_tokens=True, padding='max_length', truncation=True)
            data.append(outputs['input_ids'])
        return data

    def preprocessing(self, data:pd.DataFrame):
        # 안쓰는 컬럼을 삭제합니다.
        data = data.drop(columns=self.delete_columns)

        # 타겟 데이터가 없으면 빈 배열을 리턴합니다.
        try:
            targets = data[self.target_columns].values.tolist()
        except:
            targets = []
        # 텍스트 데이터를 전처리합니다.
        inputs = self.tokenizing(data)

        return inputs, targets

    def setup(self, stage='fit'):
        # 학습 데이터와 검증 데이터셋을 호출합니다
        train_data = pd.read_csv(self.train_path)
        val_data = pd.read_csv(self.dev_path)
        train_data = pd.concat([train_data, val_data])
        
        num = len(train_data)
        #분리 작업(전처리 후에는 문장 위치가 사라지니 미리 나눔)
        train_data, val_data, test_data = train_data.iloc[0:int(num*0.7), :], train_data.iloc[int(num*0.7):int(num*0.85), :], train_data.iloc[int(num*0.85):, :]
        
        #특수문자 제거
        for text_col in self.text_columns:
            train_data[text_col] = train_data[text_col].str.replace(pat=r'[^\w]',repl=r' ',regex=True)
            val_data[text_col] = val_data[text_col].str.replace(pat=r'[^\w]',repl=r' ',regex=True)
            test_data[text_col] = test_data[text_col].str.replace(pat=r'[^\w]',repl=r' ',regex=True)

        # 영어 모두 소문자로
        for text_col in self.text_columns:
            train_data[text_col] = train_data[text_col].str.lower()
            val_data[text_col] = val_data[text_col].str.lower()
            test_data[text_col] = test_data[text_col].str.lower()

        self.test_data = test_data
        if stage == 'fit':
            
            #데이터 증강
            train_data2 = train_data.copy()
            train_data2.rename(columns={'sentence_1':'sentence_2', 'sentence_2':'sentence_1'}, inplace = True)
            train_data = pd.concat([train_data, train_data2])
            
            data_ = train_data[train_data['label'] != 0]
            smoothing_data = train_data[train_data['label'] == 0]
            smoothing_data2 = smoothing_data.sample(frac = 0.5)
            smoothing_data3 = pd.concat([smoothing_data2, smoothing_data])
            smoothing_data3 = smoothing_data3.drop_duplicates(keep = False)
            smoothing_data3 = smoothing_data3.sample(frac = 0.5)
            label5_sentence_data = smoothing_data3['sentence_1']
            smoothing_data3['sentence_2'] = label5_sentence_data
            smoothing_data3['label'] = [5 for i in range(len(smoothing_data3))]
            smoothing_data3['binary-label'] = [1 for i in range(len(smoothing_data3))]
            train_data = pd.concat([smoothing_data2, smoothing_data3, data_])
            train_data = train_data.sample(frac = 1)
            
            # 학습데이터 준비
            train_inputs, train_targets = self.preprocessing(train_data)
            print('train_inputs:', len(train_inputs))
            print('train_targets:', len(train_targets))


            # 검증데이터 준비
            val_inputs, val_targets = self.preprocessing(val_data)
            
            # train 데이터만 shuffle을 적용해줍니다, 필요하다면 val, test 데이터에도 shuffle을 적용할 수 있습니다
            self.train_dataset = Dataset(train_inputs, train_targets)
            self.val_dataset = Dataset(val_inputs, val_targets)
        else:
            # 평가데이터 준비
            #test_data = pd.read_csv(self.test_path)
            test_inputs, test_targets = self.preprocessing(self.test_data)
            
            self.test_dataset = Dataset(test_inputs, test_targets)
            
            predict_data = pd.read_csv(self.predict_path)

            # 특수문자 제거
            for text_col in self.text_columns:
                predict_data[text_col] = predict_data[text_col].str.replace(pat=r'[^\w]',repl=r' ',regex=True)
            
            # 영어 소문자로
            for text_col in self.text_columns:
                predict_data[text_col] = predict_data[text_col].str.lower()
                
            predict_inputs, predict_targets = self.preprocessing(predict_data)
            self.predict_dataset = Dataset(predict_inputs, [])

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=self.shuffle, drop_last=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size, drop_last=True)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, drop_last=True)

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(self.predict_dataset, batch_size=self.batch_size, drop_last=True)
