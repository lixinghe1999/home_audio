import torch
import torch.nn as nn
import torch.nn.functional as F
from .model import get_model as get_mobilenet_model
from .model import NAME_TO_WIDTH
from .preprocess import AugmentMelSTFT

class Sound_Event_Detector(nn.Module):
    def __init__(self, model_name = 'mn10_as', num_classes=527, frame_duration=None):
        super().__init__()
        self.preprocess = AugmentMelSTFT()
        self.backbone = get_mobilenet_model(num_classes=num_classes, pretrained_name=model_name, width_mult=NAME_TO_WIDTH(model_name), 
                                         strides=[2, 2, 2, 2], head_type='mlp')
        if frame_duration is not None:
            self.frame_per_second = 50
            self.frame_length = int(self.frame_per_second * frame_duration)
        self.condition_classifier = torch.nn.Linear(num_classes + 512, num_classes)

    def forward(self, x, vision=None, return_fmaps=False):
        x = self.preprocess(x)
        x = x.unsqueeze(1)
        B, C, Freq, T = x.shape
        _T = T//self.frame_length
        x = x.view(B*_T, 1, Freq, self.frame_length)
        x, fmaps = self.backbone(x, return_fmaps=return_fmaps)
        if vision is not None:
            number_repeat = x.shape[0] // vision.shape[0]
            vision = torch.repeat_interleave(vision, number_repeat, dim=0).squeeze()
            x = self.condition_classifier(torch.cat([x, vision], dim=1))
        x = x.view(B, _T, -1)
        return x

class Frame_MobileNet(torch.nn.Module):
    def __init__(self, backbone, num_classes, frame_length: int = 50):
        super().__init__()
        self.backbone = backbone
        self.frame_length = frame_length
        self.condition_classifier = torch.nn.Linear(num_classes + 512, num_classes)

    def forward(self, x, vision=None, return_fmaps: bool = False):
        B, C, Freq, T = x.shape
        _T = T//self.frame_length
        x = x.view(B*_T, 1, Freq, self.frame_length)
        # x = self._forward(x, vision, return_fmaps=return_fmaps)
        x, fmaps = self.backbone(x, return_fmaps=return_fmaps)

        if vision is not None:
            number_repeat = x.shape[0] // vision.shape[0]
            vision = torch.repeat_interleave(vision, number_repeat, dim=0).squeeze()
            x = self.condition_classifier(torch.cat([x, vision], dim=1))

        x = x.view(B, _T, -1)
        return x
    