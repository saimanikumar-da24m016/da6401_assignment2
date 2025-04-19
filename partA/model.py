import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


class LitCNN(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(cfg)
        # Build conv blocks
        if cfg["filter_organization"] == 'same':
            conv_filters = [32]*5
        elif cfg["filter_organization"] == 'double':
            conv_filters = [32,64,128,256,512]
        elif cfg["filter_organization"] == 'half':
            conv_filters = [512,256,128,64,32]
        else:
            raise ValueError("Invalid filter organization")
        layers = []
        in_ch = 3
        for f in conv_filters:
            layers.append(nn.Conv2d(in_ch, f, kernel_size=3, padding=1))
            if cfg["batch_norm"]: layers.append(nn.BatchNorm2d(f))
            layers.append(getattr(nn, cfg["activation"].capitalize())())
            layers.append(nn.MaxPool2d(2))
            in_ch = f
        self.conv = nn.Sequential(*layers)
        size = cfg["img_size"] // 32
        self.fc1 = nn.Linear(conv_filters[-1]*size*size, cfg["dense_neurons"])
        self.dp = nn.Dropout(cfg["dropout"])
        self.fc2 = nn.Linear(cfg["dense_neurons"], cfg["num_classes"])

    def forward(self, x):
        x = self.conv(x)
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = self.dp(x)
        return self.fc2(x)

    def training_step(self, batch, _):
        x,y = batch; logits = self(x)
        loss = F.cross_entropy(logits,y)
        acc  = (logits.argmax(1)==y).float().mean()
        self.log('train_loss', loss); self.log('train_acc', acc)
        return loss

    def validation_step(self, batch, _):
        x,y = batch; logits = self(x)
        loss = F.cross_entropy(logits,y)
        acc  = (logits.argmax(1)==y).float().mean()
        self.log('val_loss', loss, prog_bar=True); self.log('val_acc', acc, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams['learning_rate'])
