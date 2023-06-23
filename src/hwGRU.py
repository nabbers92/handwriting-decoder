import os
from argparse import ArgumentParser

import numpy as np
import pytorch_lightning as pl
import scipy
import torch
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.utilities.grads import grad_norm
from torch import nn
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import DataLoader, Dataset, Subset, random_split

torch.set_float32_matmul_precision('high')

"""
TODO:
Account for 50 timebin delay in training (see github for example)
Update second layer only every five timesteps, upsample to fill in remaining timesteps
Implement day-specific sampling of data, both synthetic and real
Implement error weights
Implement specifying ratio of real data to synthetic data
Implement noise additions to data
Implement day-specific affine layers
"""


def zScore(neuralCube):
    """
    The neural cube has dimensions B x L X E
    where B is the batch size, L is the sequence length, and E is the electrode
    count
    """
    channelMean = torch.mean(neuralCube, dim=1)
    zScoreCube = neuralCube - channelMean[:, None, :]
    binStd = torch.std(neuralCube, dim=2)
    idx = torch.where(binStd == 0)
    binStd[idx] = 1
    zScoreCube = zScoreCube / binStd[:, :, None]
    return zScoreCube


class HandwritingDataset(Dataset):
    def __init__(self, x, y, z):
        self.x = x.clone().detach()
        self.y = y.clone().detach()
        self.z = z.clone().detach()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index], self.z[index]


class HandwritingDataModuleWithSynth(pl.LightningDataModule):
    def __init__(self, data_dir: str = "data/handwritingData",
                 batch_size: int = 64):
        super(HandwritingDataModuleWithSynth, self).__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

    def setup(self, stage: str):
        # initializing empty arrays to append the data
        self.x = np.empty((1, 1200, 192))
        self.y = np.empty((1, 1200, 31))
        self.z = np.empty((1, 1200, 1))
        for filename in os.listdir(self.data_dir):
            if filename.endswith(".pt"):
                file_path = os.path.join(self.data_dir, filename)

                data = torch.load(file_path)
                x = data['inputs']
                y = data['charLabels']
                z = data['charStarts']
                self.x = np.concatenate((self.x, x), axis=0)
                self.y = np.concatenate((self.y, y), axis=0)
                self.z = np.concatenate((self.z, z), axis=0)

                # remove empty initialization index
                self.x = self.x[1:, :, :]
                self.y = self.y[1:, :, :]
                self.z = self.z[1:, :, :]

                # change to torch tensor
                self.x = torch.tensor(self.x, dtype=torch.float)
                self.y = torch.tensor(self.y, dtype=torch.float)
                self.z = torch.tensor(self.z, dtype=torch.float)

        # z-scoring and smoothing spike input before converting to torch tensor
        self.x = zScore(self.x)
        self.x = scipy.ndimage.gaussian_filter1d(self.x, 2.0, axis=1)
        self.x = torch.tensor(self.x, dtype=torch.float)
        data_full = HandwritingDataset(self.x, self.y, self.z)

        if stage == "fit":
            self.train_dataset, self.val_dataset = random_split(data_full,
                                                                [0.9, 0.1])

        if stage == "test":
            self.test_dataset = Subset(data_full, list(range(250)))

    def train_dataloader(self):
        return DataLoader(self.train_dataset, shuffle=True,
                          batch_size=self.batch_size, num_workers=24)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, shuffle=False,
                          batch_size=self.batch_size, num_workers=24)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, shuffle=False,
                          batch_size=1, num_workers=24)


class HandwritingGRU(pl.LightningModule):
    def __init__(self, input_size, hidden_size, num_chars, reg_strength, lr):
        super(HandwritingGRU, self).__init__()
        self.save_hyperparameters()
        self.affine = nn.Linear(input_size, input_size)
        self.gru1 = nn.GRU(input_size, hidden_size, batch_first=True)
        self.gru2 = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.fc_y = nn.Linear(hidden_size, num_chars)
        self.fc_z = nn.Linear(hidden_size, 1)
        self.ce_loss = nn.CrossEntropyLoss(reduction='mean')
        self.mse_loss = nn.MSELoss(reduction='mean')
        self.hidden_size = hidden_size
        self.reg_strength = reg_strength
        self.learning_rate = lr

    def forward(self, x):
        x = self.affine(x)
        out, h = self.gru1(x)
        out, _ = self.gru2(out, h)
        y_logits = self.fc_y(out)
        z_logits = self.fc_z(out)
        zhat = torch.sigmoid(z_logits)
        return y_logits, zhat, h

    def training_step(self, batch, batch_idx):
        x, y, z = batch

        y_logits, zhat, _ = self(x)
        y_logits = torch.transpose(y_logits, 1, 2)

        y = y.argmax(dim=2)

        ce_loss = self.ce_loss(y_logits, y)
        mse_loss = self.mse_loss(zhat, z)

        # Determine L2 regularization terms to use for loss
        norm_w_affine = torch.norm(self.affine.weight, p='fro')
        norm_w_gru1 = torch.norm(
            self.gru1.weight_ih_l0, p='fro') + torch.norm(self.gru1.weight_hh_l0, p='fro')
        norm_w_gru2 = torch.norm(
            self.gru2.weight_ih_l0, p='fro') + torch.norm(self.gru2.weight_hh_l0, p='fro')
        norm_w_fc_y = torch.norm(self.fc_y.weight, p='fro')
        norm_w_fc_z = torch.norm(self.fc_z.weight, p='fro')
        l2_loss = (norm_w_affine**2 + norm_w_gru1**2 + norm_w_gru2**2 +
                   norm_w_fc_y**2 + norm_w_fc_z**2)
        loss = self.reg_strength*l2_loss + ce_loss + mse_loss

        accuracy = torch.sum(y_logits.argmax(dim=1) == y)/y.shape[1]

        self.log('l2_loss', self.reg_strength*l2_loss, on_step=True)
        self.log('ce_loss', ce_loss, on_step=True)
        self.log('mse_loss', mse_loss, on_step=True)
        self.log('training_loss', loss, on_step=True, prog_bar=True)
        self.log('accuracy', accuracy, on_step=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y, z = batch

        y_logits, zhat, _ = self(x)
        y_logits = torch.transpose(y_logits, 1, 2)
        y = y.argmax(dim=2)

        ce_loss = self.ce_loss(y_logits, y)
        mse_loss = self.mse_loss(zhat, z)

        # Determine L2 regularization terms to use for loss
        norm_w_affine = torch.norm(self.affine.weight, p='fro')
        norm_w_gru1 = torch.norm(
            self.gru1.weight_ih_l0, p='fro') + torch.norm(self.gru1.weight_hh_l0, p='fro')
        norm_w_gru2 = torch.norm(
            self.gru2.weight_ih_l0, p='fro') + torch.norm(self.gru2.weight_hh_l0, p='fro')
        norm_w_fc_y = torch.norm(self.fc_y.weight, p='fro')
        norm_w_fc_z = torch.norm(self.fc_z.weight, p='fro')
        l2_loss = (norm_w_affine**2 + norm_w_gru1**2 + norm_w_gru2**2 +
                   norm_w_fc_y**2 + norm_w_fc_z**2)

        loss = self.reg_strength*l2_loss + ce_loss + mse_loss

        self.log('validation_loss', loss, on_step=True)

    def test_step(self, batch, batch_idx):
        x, y, z = batch

        y_logits, zhat, _ = self(x)
        y_logits = torch.transpose(y_logits, 1, 2)
        y = y.argmax(dim=2)

        ce_loss = self.ce_loss(y_logits, y)
        mse_loss = self.mse_loss(zhat, z)

        # Determine L2 regularization terms to use for loss
        norm_w_affine = torch.norm(self.affine.weight, p='fro')
        norm_w_gru1 = torch.norm(
            self.gru1.weight_ih_l0, p='fro') + torch.norm(self.gru1.weight_hh_l0, p='fro')
        norm_w_gru2 = torch.norm(
            self.gru2.weight_ih_l0, p='fro') + torch.norm(self.gru2.weight_hh_l0, p='fro')
        norm_w_fc_y = torch.norm(self.fc_y.weight, p='fro')
        norm_w_fc_z = torch.norm(self.fc_z.weight, p='fro')
        l2_loss = (norm_w_affine**2 + norm_w_gru1**2 + norm_w_gru2**2 +
                   norm_w_fc_y**2 + norm_w_fc_z**2)
        loss = self.reg_strength*l2_loss + ce_loss + mse_loss
        self.log('test_loss', loss, on_step=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": LinearLR(optimizer, start_factor=1.0, end_factor=0.0,
                                      total_iters=500, verbose=True),
            }
        }

    def on_before_optimizer_step(self, optimizer):
        # inspect (unscaled) gradients here
        self.log_dict(grad_norm(self, norm_type=2))


def main(hparams):
    model = HandwritingGRU(input_size=192, hidden_size=512, num_chars=31,
                           reg_strength=0.001, lr=0.01)
    dm = HandwritingDataModuleWithSynth(
        data_dir="data/trainingData", batch_size=64)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    trainer = pl.Trainer(accelerator=hparams.accelerator,
                         devices=hparams.devices, gradient_clip_val=10,
                         max_epochs=hparams.epochs, detect_anomaly=True,
                         callbacks=[lr_monitor])
    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--accelerator', default='gpu')
    parser.add_argument('--devices', default=1)
    parser.add_argument('--epochs', default=10, type=int)
    args = parser.parse_args()

    main(args)
