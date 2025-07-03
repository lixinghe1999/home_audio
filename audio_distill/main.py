import pytorch_lightning as pl
from utils.recognition import AudioRecognition
if __name__ == "__main__":
    import json
    config = json.load(open('config.json', 'r'))
    logger = pl.loggers.CSVLogger(save_dir='./',)
    trainer = pl.Trainer(max_epochs=3, devices=[0], logger=logger)
    model = AudioRecognition(config)

    if config['pretrained']:
        ckpt = config['pretrained']
        model = AudioRecognition.load_from_checkpoint(ckpt, config=config)
        print('loaded pretrained model from', ckpt)
    else:
        print('training from scratch')

    if config['train']:
        trainer.fit(model)
    else:
        trainer.validate(model)