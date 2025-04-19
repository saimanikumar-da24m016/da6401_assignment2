import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets, models
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import wandb


def parse_args():
    parser = argparse.ArgumentParser(description="Modular Fine-tuning with EfficientNetV2")
    parser.add_argument('--config', type=str, help='Path to JSON config file')
    parser.add_argument('--mode', choices=['sweep','prog'], default='sweep',
                        help='Run mode: sweep or progressive unfreeze')
    # hyperparameters
    parser.add_argument('--img_size', type=int)
    parser.add_argument('--num_classes', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--epochs_per_stage', type=int)
    parser.add_argument('--data_augmentation', type=lambda x: x.lower()=='true')
    parser.add_argument('--freeze_before', type=int)
    parser.add_argument('--dropout', type=float)
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--sweep_count', type=int, default=10)
    parser.add_argument('--project', type=str, default='da6401_assignment2')
    return parser.parse_args()



def load_config(args):
    # default config
    cfg = {
        'img_size': 224,
        'num_classes': 10,
        'batch_size': 32,
        'epochs': 10,
        'epochs_per_stage': 5,
        'data_augmentation': True,
        'freeze_before': 7,
        'dropout': 0.3,
        'learning_rate': 1e-4
    }
    if args.config:
        with open(args.config) as f:
            file_cfg = json.load(f)
        cfg.update(file_cfg)
    # override with CLI args
    for key in ['img_size','num_classes','batch_size','epochs','epochs_per_stage',
                'data_augmentation','freeze_before','dropout','learning_rate']:
        val = getattr(args, key)
        if val is not None:
            cfg[key] = val
    return cfg





def prepare_data(cfg):
    train_tf = transforms.Compose([
        transforms.Resize((cfg['img_size'], cfg['img_size'])),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ]) if cfg['data_augmentation'] else transforms.Compose([
        transforms.Resize((cfg['img_size'], cfg['img_size'])),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    val_tf = transforms.Compose([
        transforms.Resize((cfg['img_size'], cfg['img_size'])),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    train_path = os.getenv('TRAIN_DIR','/kaggle/input/inaturalist_12K/train')
    val_path = os.getenv('VAL_DIR','/kaggle/input/inaturalist_12K/val')
    # load and stratify
    full = datasets.ImageFolder(train_path, transform=train_tf)
    idxs_by_cls = {}
    for idx,(_,lbl) in enumerate(full.samples): idxs_by_cls.setdefault(lbl,[]).append(idx)
    train_idx, val_idx = [], []
    for lbl, idxs in idxs_by_cls.items():
        np.random.shuffle(idxs)
        split = int(0.8*len(idxs))
        train_idx += idxs[:split]
        val_idx   += idxs[split:]
    train_ds = Subset(full, train_idx)
    full.transform = val_tf
    val_ds = Subset(full, val_idx)
    train_loader = DataLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_ds, batch_size=cfg['batch_size'], shuffle=False, num_workers=4)
    test_ds      = datasets.ImageFolder(val_path, transform=val_tf)
    test_loader  = DataLoader(test_ds, batch_size=cfg['batch_size'], shuffle=False, num_workers=4)
    return train_loader, val_loader, test_loader, test_ds


#### model definition class
def get_model(cfg):
    class Model(pl.LightningModule):
        def __init__(self):
            super().__init__()
            self.save_hyperparameters(cfg)
            self.net = models.efficientnet_v2_s(pretrained=True)
            total = len(self.net.features)
            freeze_upto = total - cfg['freeze_before']
            for idx,blk in enumerate(self.net.features):
                for p in blk.parameters(): p.requires_grad = (idx>=freeze_upto)
            in_f = self.net.classifier[1].in_features
            self.net.classifier = nn.Sequential(
                nn.Dropout(cfg['dropout']),
                nn.Linear(in_f, cfg['num_classes'])
            )
        def forward(self,x): return self.net(x)
        def step(self,b,tag):
            x,y=b; logits=self(x)
            loss=F.cross_entropy(logits,y)
            acc=(logits.argmax(1)==y).float().mean()
            self.log(f"{tag}_loss",loss,prog_bar=True)
            self.log(f"{tag}_acc", acc, prog_bar=True)
            return loss
        def training_step(self,b,i): return self.step(b,'train')
        def validation_step(self,b,i): return self.step(b,'val')
        def test_step(self,b,i): return self.step(b,'test')
        def configure_optimizers(self):
            params = filter(lambda p:p.requires_grad,self.parameters())
            return torch.optim.Adam(params, lr=cfg['learning_rate'])
    return Model()




def run_sweep(cfg, args):
    sweep_id = wandb.sweep({
        'method':'bayes',
        'metric':{'name':'val_acc','goal':'maximize'},
        'parameters':{
            'freeze_before':{'values':[5,7,9]},
            'dropout':{'values':[0.2,0.3]},
            'learning_rate':{'values':[1e-3,1e-4]}
        }
    }, project=args.project)
    def _train():
        run=wandb.init()
        p = {**cfg, **run.config}
        run.name=f"freeze{p['freeze_before']}_do{p['dropout']}_lr{p['learning_rate']}"
        train_loader,val_loader,test_loader,_ = prepare_data(p)
        model = get_model(p)
        wandb_logger=WandbLogger(project=args.project)
        ckpt=ModelCheckpoint(monitor='val_acc',mode='max',save_top_k=1)
        trainer=pl.Trainer(max_epochs=p['epochs'],logger=wandb_logger,callbacks=[ckpt],accelerator='gpu')
        trainer.fit(model,train_loader,val_loader)
        trainer.test(model,test_loader)
        wandb.finish()
    wandb.agent(sweep_id, function=_train, count=args.sweep_count)

def run_progressive(cfg,args):
    train_loader,val_loader,test_loader,_ = prepare_data(cfg)
    model = get_model(cfg)
    wandb.init(project=args.project,name='progressive_unfreeze')
    wandb_logger=WandbLogger(project=args.project)
    ckpt=ModelCheckpoint(monitor='val_acc',mode='max',save_top_k=1)
    total = len(model.net.features)
    stages=[0,1,2,total]
    for idx,u in enumerate(stages):
        # unfreeze last u blocks
        for i,blk in enumerate(model.net.features):
            for p in blk.parameters(): p.requires_grad = (i>= total-u)
        print(f"Stage {idx+1}: unfreeze_last={u}")
        trainer=pl.Trainer(max_epochs=cfg['epochs_per_stage'],logger=wandb_logger,callbacks=[ckpt],accelerator='gpu')
        trainer.fit(model,train_loader,val_loader)
    # test
    best = ckpt.best_model_path
    print("Best ckpt:",best)
    m = get_model(cfg)
    m = m.load_from_checkpoint(best)
    m.eval(); m.to('cuda')
    trainer = pl.Trainer(accelerator='gpu')
    trainer.test(m,test_loader)
    wandb.finish()

def main():
    args = parse_args()
    cfg = load_config(args)
    if args.mode=='sweep':
        run_sweep(cfg,args)
    else:
        run_progressive(cfg,args)

## ways we can run this script:
# python partB_train.py --mode sweep --sweep_count 10
# python partB_train.py --mode prog --config best_params.json

if __name__=='__main__':
    main()
