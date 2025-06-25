import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,3'
import time
import argparse
import json

import torch
import pytorch_lightning as pl

from config import Config
from dataset import PaviaUDataset
from model import LineRWKV
from trainer import train_callback


parser = argparse.ArgumentParser()
parser.add_argument('--log_dir', default='log_dir', help='Tensorboard log directory')
parser.add_argument('--save_dir', default='Results', help='Trained model directory')
param = parser.parse_args()

model_time = time.strftime("%Y%m%d_%H%M")


# Import config
config = Config()
config.save_dir = os.path.join(param.save_dir, str(model_time))

# Import datasets and validation image

# PaviaU dataset
train_dataset = PaviaUDataset(config, mode ='train')
val_dataset = PaviaUDataset(config, mode='val')
test_dataset = PaviaUDataset(config, mode='test')

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, num_workers=config.num_workers, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=4, num_workers=config.num_workers,shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)


# Create Tensorboard logger
log_dir = os.path.join(param.log_dir, model_time)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
with open(os.path.join(log_dir,'config.txt'), 'w') as cfg_file:
    cfg_file.write(json.dumps(config, default=lambda obj: obj.__dict__))
logger = pl.loggers.TensorBoardLogger(param.log_dir, name=model_time)


# Create save directory
save_dir = os.path.join(param.save_dir, model_time)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Create model and train
model = LineRWKV(config)
accelerator = "gpu" if config.device=="cuda" else "cpu"
#trainer = pl.Trainer(accelerator=accelerator, logger=logger, callbacks=[train_callback(config)])
trainer = pl.Trainer(accelerator=accelerator, devices=1, default_root_dir=save_dir, logger=logger, max_epochs=config.epoch_count, callbacks=[train_callback(config)])
trainer.fit(model, train_loader, val_loader)

