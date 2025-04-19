"""
Modular PyTorch Lightning training script for iNaturalist CNN & fine-tuning.
the __main__ entry  accepts a config dict for training...
"""
import os
import json
import argparse
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader, Subset
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import wandb
from .model import LitCNN


def get_config(config_path=None, overrides=None):
    # default hyperparameters for the model.
    default = {
        "img_size": 128,
        "num_classes": 10,
        "batch_size": 32,
        "epochs": 10,
        "filter_organization": "double",
        "activation": "relu",
        "data_augmentation": True,
        "batch_norm": True,
        "dropout": 0.2,
        "dense_neurons": 256,
        "learning_rate": 1e-3
    }
    if config_path:
        with open(config_path) as f:
            cfg = json.load(f)
        default.update(cfg)
    if overrides:
        default.update(overrides)
    return default


def prepare_data(cfg):
    # transformations
    train_tf = transforms.Compose([
        transforms.Resize((cfg["img_size"], cfg["img_size"])),
        transforms.RandomHorizontalFlip() if cfg["data_augmentation"] else transforms.Lambda(lambda x: x),
        transforms.RandomRotation(15) if cfg["data_augmentation"] else transforms.Lambda(lambda x: x),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    val_tf = transforms.Compose([
        transforms.Resize((cfg["img_size"], cfg["img_size"])),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    train_path = os.environ.get("TRAIN_PATH", "./train")
    val_path = os.environ.get("VAL_PATH", "./val")

    full = datasets.ImageFolder(train_path, transform=train_tf)
    idxs_by_cls = defaultdict(list)
    for idx, (_, lbl) in enumerate(full.samples): idxs_by_cls[lbl].append(idx)
    train_idx, val_idx = [], []
    for lbl, idxs in idxs_by_cls.items():
        np.random.shuffle(idxs)
        cut = int(0.8 * len(idxs))
        train_idx += idxs[:cut]
        val_idx += idxs[cut:]
    train_ds = Subset(full, train_idx)
    full.transform = val_tf
    val_ds = Subset(full, val_idx)
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=4)
    return train_loader, val_loader


def main(config):
    wandb.init(project="da6401_assignment2", config=config)
    cfg = wandb.config
    train_loader, val_loader = prepare_data(cfg)
    logger = WandbLogger()
    ckpt = ModelCheckpoint(monitor='val_acc', mode='max', save_top_k=1)
    trainer = pl.Trainer(max_epochs=cfg.epochs, logger=logger, callbacks=[ckpt], devices=1, accelerator='gpu')
    model = LitCNN(cfg)
    trainer.fit(model, train_loader, val_loader)
    wandb.finish()
    
    
# Example of how to run the script from the command line:
# python train_partA_parser.py --config config.json
# --img_size
# --num_classes
# --batch_size
# --epochs
# --filter_organization
# --activation
# --data_augmentation
# --batch_norm
# --dropout
# --dense_neurons
# --learning_rate


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Path to JSON config', default=None)
    args = parser.parse_args()
    cfg = get_config(args.config)
    main(cfg)
