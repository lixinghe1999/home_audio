import pytorch_lightning as pl
from .recognition import AudioRecognition
import torch

class DistillModel(AudioRecognition):
    def __init__(self, config):
        super().__init__(config)
        self.teacher = AudioRecognition(config)
        self.teacher.load_from_checkpoint(config['teacher_ckpt'], config=config)
        self.teacher.eval()
        self.teacher.freeze()  # Freeze the teacher model
        self.student = AudioRecognition(config)  # Initialize the student model

    def training_step(self, batch, batch_idx):
        x = [batch[k] for k in self.config['modality']]
        y = batch['cls_label']
        y_hat = self.student(x)
        loss = self.loss(y_hat, y)

        # Get the teacher's predictions
        with torch.no_grad():
            teacher_y_hat = self.teacher(x)
        loss_kl = self.kl_loss(y_hat, teacher_y_hat)
        loss = loss + self.config['alpha'] * loss_kl
        self.log('loss', loss, on_step=True, prog_bar=True, logger=True)   
        return loss
    
    def test_step(self, batch, batch_idx):
        x = [batch[k] for k in self.config['modality']]
        y = batch['cls_label']
        y_hat = self.model(x)
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