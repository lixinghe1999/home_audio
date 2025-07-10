'''
Audio recognition script
'''
import pytorch_lightning as pl
from utils.EfficientAT.frame_mn import Sound_Event_Detector
from utils.recognition_dataset import AudioSet_dataset, FSD50K_dataset
from torch.utils.data import random_split, Subset
import torch
import torchmetrics
import os 
import warnings
warnings.filterwarnings("ignore")

class AudioRecognition(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
    
        if self.config['dataset'] == 'audioset': # load default dataset
            self.train_dataset = AudioSet_dataset(root=config['dataset_folder'], split='train', frame_duration=self.config['frame_duration'], 
                                                    modality=self.config['modality'], label_level=self.config['label_level'], 
                                                    num_inherit=self.config['inherit'], cls_filter=self.config['cls_filter'])
            vocabulary = self.train_dataset.vocabulary; self.num_classes = self.train_dataset.num_classes
            self.test_dataset = AudioSet_dataset(root=config['dataset_folder'], split='eval', frame_duration=self.config['frame_duration'],
                                modality=self.config['modality'], label_level=self.config['label_level'], num_inherit=self.config['inherit'], 
                                vocabulary=vocabulary, cls_filter=self.config['cls_filter'])
            self.config['vocabulary'] = self.train_dataset.idx2name
        else:
            raise ValueError('dataset not supported')
        self.config['num_classes'] = self.num_classes
        self.config['num_train_samples'] = len(self.train_dataset)
        self.config['num_test_samples'] = len(self.test_dataset)
        self.model = Sound_Event_Detector(self.config['model_name'], self.num_classes, 
                                    frame_duration=self.config['frame_duration'] if self.config['label_level'] == 'frame' else None,)

        if self.config['freeze_backbone']:
            for param in self.model.backbone.parameters():
                param.requires_grad = False
            print('backbone frozen')

        if self.config['task'] == 'multiclass':
            self.accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=self.num_classes)
        elif self.config['task'] == 'multilabel':
            self.accuracy = torchmetrics.AveragePrecision(task='multilabel', num_labels=self.num_classes, average='micro')
        else:
            raise ValueError('task not supported')
        # save the config
        self.save_hyperparameters(config)

    def training_step(self, batch, batch_idx):
        x = [batch[k] for k in self.config['modality']]
        y = batch['cls_label']
        y_hat = self.model(*x)
        loss = self.loss(y_hat, y)
        self.log('loss', loss, on_step=True, prog_bar=True, logger=True)   
        return loss
    
    def validation_step(self, batch, batch_idx):
        x = [batch[k] for k in self.config['modality']]
        y = batch['cls_label']
        y_hat = self.model(*x)
        assert y_hat.shape == y.shape
        if len(y_hat.shape) == 3:
            y_hat = y_hat.reshape(-1, y_hat.shape[-1])
            y = y.reshape(-1, y.shape[-1])
        y_hat = torch.sigmoid(y_hat)
        val_acc = self.accuracy(y_hat.cpu(), y.cpu().long())
        if torch.isnan(val_acc):
            print('nan detected, no positive samples')
        else:
            self.log('validation', val_acc, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.config['lr'])

    def loss(self, y_hat, y):
        # single-label classification
        if self.config['task'] == 'multiclass':
            return torch.nn.CrossEntropyLoss()(y_hat, y)
        elif self.config['task'] == 'multilabel':
            return torch.nn.BCEWithLogitsLoss()(y_hat, y.float())
        else:
            raise ValueError('task not supported')
    
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.config['batch_size'], shuffle=True, num_workers=self.config['num_workers'], pin_memory=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.config['batch_size'], shuffle=False, num_workers=self.config['num_workers'])  

