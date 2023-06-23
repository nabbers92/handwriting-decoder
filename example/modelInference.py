from utils.characterDefinitions import getHandwritingCharacterDefinitions
import os
import torch
import numpy as np
import scipy
import matplotlib.pyplot as plt
from hwGRU import HandwritingGRU, HandwritingDataset
torch.set_float32_matmul_precision('high')


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


data_dir = os.path.expanduser('~') + '/handwriting-model/data/testingData'

x = np.empty((1, 1200, 192))
y = np.empty((1, 1200, 31))
z = np.empty((1, 1200, 1))
for filename in os.listdir(data_dir):
    if filename.endswith(".pt"):
        file_path = os.path.join(data_dir, filename)

        data = torch.load(file_path)
        x_temp = data['inputs']
        # print(x.shape)
        y_temp = data['charLabels']
        z_temp = data['charStarts']
        x = np.concatenate((x, x_temp), axis=0)
        y = np.concatenate((y, y_temp), axis=0)
        z = np.concatenate((z, z_temp), axis=0)
print(x.shape)

# remove empty initialization index
x = x[1:, :, :]
y = y[1:, :, :]
z = z[1:, :, :]

# change to torch tensor
x = torch.tensor(x, dtype=torch.float)
y = torch.tensor(y, dtype=torch.float)
z = torch.tensor(z, dtype=torch.float)

# z-scoring and smoothing spike input before converting to torch tensor
x = zScore(x)
x = scipy.ndimage.gaussian_filter1d(x, 2.0, axis=1)
x = torch.tensor(x, dtype=torch.float)
data_full = HandwritingDataset(x, y, z)

#baseDir = os.path.expanduser('~') + '/handwriting-model/lightning_logs'
#version = 1
#ckptDir = os.path.join(baseDir, f'version_{version}', 'checkpoints')
#ckptFile = os.path.join(ckptDir, os.listdir(ckptDir)[0])
ckptFile = 'data/epoch=499-step=105500.ckpt'
model = HandwritingGRU.load_from_checkpoint(ckptFile)
print(f"Device: {model.device}")
device = model.device
model.eval()

charDef = getHandwritingCharacterDefinitions()
chars = charDef['charListAbbr']
threshold = 0.4
total_accuracies = []
for n in range(len(data_full)):
    x, y, z = data_full[n]
    x, y, z = x.to(device), y.to(device), z.to(device)
    y_logits, zhat, _ = model(x)

    trial_accuracy = y.clone().detach().argmax(
        dim=1) == y_logits.clone().detach().argmax(dim=1)
    trial_accuracy = trial_accuracy.int()
    trial_accuracy = torch.sum(trial_accuracy)
    total_accuracies.append(trial_accuracy.cpu()/1200)

    zhat = zhat.cpu()
    zhat = zhat.detach().numpy()
    zhat = np.squeeze(zhat)

    z = z.cpu()
    z = z.detach().numpy()
    z = np.squeeze(z)

    # yhat = torch.softmax(y_logits, dim=2)
    # yhat = torch.softmax(y_logits, dim=1)  # for use if sampling directly from set
    yhat = y_logits
    yhat = yhat.cpu()
    yhat = yhat.detach().numpy()
    yhat = np.squeeze(yhat)

    y = y.cpu()
    y = y.detach().numpy()
    y = np.squeeze(y)

    if n % 500 == 0:
        predicted_crossings = np.diff(zhat > threshold)
        predicted_crossings = np.argwhere(predicted_crossings)[
            ::2, 0].astype(int)
        predicted_chars = yhat.T[:, predicted_crossings].argmax(axis=0)

        actual_crossings = np.diff(z > threshold)
        actual_crossings = np.argwhere(actual_crossings)[::2, 0].astype(int)
        actual_chars = y.T[:, actual_crossings].argmax(axis=0)

        print(f"Actual Characters: {np.take(chars, actual_chars)}")
        print(f"Predicted Characters: {np.take(chars, predicted_chars)}")
        print(f"Frame-by-Frame Accuracy: {trial_accuracy*100/1200:.2f}%")

        fig, ax = plt.subplots(2, 2)
        fig.set_size_inches(18.5, 10.5)
        ax[0][0].plot(z)
        ax[0][0].set_xlim([0, 1200])
        ax[0][0].set_title("Target")
        ax[0][0].set_ylabel('New Character Start Probability')
        ax[0][1].plot(zhat)
        ax[0][1].set_title("Prediction")
        ax[0][1].set_xlim([0, 1200])
        ax[1][0].imshow(y.T, aspect='auto')
        ax[1][0].set_xlim([0, 1200])
        ax[1][0].set_xlabel("Time Bin (20 ms)")
        ax[1][0].set_ylabel("Character")
        ax[1][1].imshow(yhat.T, aspect='auto')
        ax[1][1].set_xlim([0, 1200])
        ax[1][1].set_xlabel("Time Bin (20 ms)")
        plt.show()

total_accuracy = np.array(total_accuracies)
mean_accuracy = np.mean(total_accuracy)
accuracy_std = np.std(total_accuracy)
