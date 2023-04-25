import pytorch_lightning as pl
import transformers
import torch
import torchmetrics
from torch.optim.lr_scheduler import StepLR

#피어슨 상관계수 함수
class PLCCLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output, target):
        vx = output - torch.mean(output)
        vy = target - torch.mean(target)

        corr = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2))
                                     * torch.sqrt(torch.sum(vy ** 2)))
        return 1 - corr**2

class Model(pl.LightningModule):
    def __init__(
            self, 
            model_name: str, 
            lr: float,
            weight_decay: float,
            loss: str,
        ):
        super().__init__()
        self.save_hyperparameters()

        self.model_name = model_name
        self.lr = lr
        self.loss = {
            "L1": torch.nn.L1Loss,
            "MSE": torch.nn.MSELoss,
            "Huber": torch.nn.HuberLoss,
            "plcc": PLCCLoss,
        }
        self.loss_func = self.loss[loss]()
        self.weight_decay = weight_decay

        # 사용할 모델을 호출합니다.
        self.plm = transformers.AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=model_name, num_labels=1)

        
        #Huber Loss
        #self.loss_func = torch.nn.SmoothL1Loss()
        
        # wandb 에 하이퍼파라미터 저장
        self.save_hyperparameters()

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
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = StepLR(optimizer, step_size=1)
        return [optimizer], [scheduler]
