import pytorch_lightning as pl
from utils.recognition import AudioRecognition
if __name__ == "__main__":
    config = {
        'model_name': 'mn01_as',
        'frame_duration': 1,
        'label_level': 'frame',
        'modality': ['audio', 'embeddings'],
        'inherit': 2,
        'dataset': 'audioset',
        'task': 'multilabel',
        'cls_filter': [], # ['/m/09x0r', '/m/0bt9lr'],
        'freeze_backbone': False,

        'pretrained': None,
        'train': True,
        'batch_size': 32,
        'num_workers': 8,
        'lr': 1e-3,
    }
    logger = pl.loggers.CSVLogger(save_dir='./',)
    trainer = pl.Trainer(max_epochs=3, devices=[0], logger=logger)
    model = AudioRecognition(config)

    if config['pretrained'] is not None:
        ckpt = config['pretrained']
        model = AudioRecognition.load_from_checkpoint(ckpt, config=config)
        print('loaded pretrained model from', ckpt)
    else:
        print('training from scratch')

    if config['train']:
        trainer.fit(model)
    else:
        trainer.validate(model)