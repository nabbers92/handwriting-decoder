import os
import torch
import pytorch_lightning as pl
from hwGRU import HandwritingGRU, HandwritingDataModuleWithSynth
torch.set_float32_matmul_precision('high')

model = HandwritingGRU(input_size=192, hidden_size=512, num_chars=31,
                       reg_strength=0.001, lr=0.01)

dataDir = os.getcwd() + '/data/trainingData'
dm = HandwritingDataModuleWithSynth(data_dir=dataDir, batch_size=64)

trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=1000,
                     gradient_clip_val=10, detect_anomaly=True)
trainer.fit(model, datamodule=dm)
