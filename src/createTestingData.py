import datetime
import multiprocessing
import os

import numpy as np
import scipy
import scipy.io
import torch

from utils.characterDefinitions import getHandwritingCharacterDefinitions
from utils.dataPreprocessing import normalizeSentenceDataCube
from utils.makeSyntheticSentences import (addSingleLetterSnippets,
                                          extractCharacterSnippets,
                                          generateCharacterSequences)

results = []
# create save directory
baseDir = '../example/data/testingData/'
if not os.path.isdir(baseDir):
    os.mkdir(baseDir)
# point this towards the top level dataset directory
rootDir = '../handwritingBCIData/'

# define which datasets to process
dataDirs = ['t5.2019.05.08', 't5.2019.11.25', 't5.2019.12.09', 't5.2019.12.11', 't5.2019.12.18',
            't5.2019.12.20', 't5.2020.01.06', 't5.2020.01.08', 't5.2020.01.13', 't5.2020.01.15']

# defines the list of all 31 characters and what to call them
charDef = getHandwritingCharacterDefinitions()

sentenceLength = 2400

# construct synthetic data for both training partitions
cvParts = ['HeldOutBlocks', 'HeldOutTrials']
print('Creating synthetic snippets')
for dataDir in dataDirs:
    print('Processing ' + dataDir)

    for cvPart in cvParts:
        print('--' + cvPart)

        # load datasets and train/test partition
        sentenceDat = scipy.io.loadmat(
            rootDir+'Datasets/'+dataDir+'/sentences.mat')
        singleLetterDat = scipy.io.loadmat(
            rootDir+'Datasets/'+dataDir+'/singleLetters.mat')
        twCubes = scipy.io.loadmat(
            rootDir+'RNNTrainingSteps/Step1_TimeWarping/'+dataDir+'_warpedCubes.mat')

        cvPartFile = scipy.io.loadmat(
            rootDir+'RNNTrainingSteps/trainTestPartitions_'+cvPart+'.mat')
        trainPartitionIdx = cvPartFile[dataDir+'_train']

        # the last two sessions have hashmarks (#) to indicate that T5 should take a brief pause
        # here we remove these from the sentence prompts, otherwise the code below will get confused (because # isn't a character)
        for x in range(sentenceDat['sentencePrompt'].shape[0]):
            sentenceDat['sentencePrompt'][x,
                                          0][0] = sentenceDat['sentencePrompt'][x, 0][0].replace('#', '')

        # normalize the neural activity cube
        neuralCube = normalizeSentenceDataCube(sentenceDat, singleLetterDat)

        # load labels
        labels = scipy.io.loadmat(
            rootDir + 'RNNTrainingSteps/Step2_HMMLabels/'+cvPart+'/'+dataDir+'_timeSeriesLabels.mat')

        # cut out character snippets from the data for augmentation
        snippetDict = extractCharacterSnippets(labels['letterStarts'],
                                               labels['blankWindows'],
                                               neuralCube,
                                               sentenceDat['sentencePrompt'][:, 0],
                                               sentenceDat['numTimeBinsPerSentence'][:, 0],
                                               trainPartitionIdx,
                                               charDef)

        # add single letter examples
        snippetDict = addSingleLetterSnippets(snippetDict,
                                              singleLetterDat,
                                              twCubes,
                                              charDef)

        # save results
        if not os.path.isdir(rootDir + 'RNNTrainingSteps/Step3_SyntheticSentences/'+cvPart):
            os.mkdir(rootDir + 'RNNTrainingSteps/Step3_SyntheticSentences/'+cvPart)
        scipy.io.savemat(rootDir + 'RNNTrainingSteps/Step3_SyntheticSentences/' +
                         cvPart+'/'+dataDir+'_snippets.mat', snippetDict)

print(f'\nCreating synthetic sentences')
nParallelProcesses = 10
for dataDir in dataDirs:
    print('Processing ' + dataDir)

    for cvPart in cvParts:
        print('--' + cvPart)

        outputDir = rootDir+'RNNTrainingSteps/Step3_SyntheticSentences/' + \
            cvPart+'/'+dataDir+'_syntheticSentences'
        bashDir = rootDir+'bashScratch'
        repoDir = os.path.expanduser(
            '~') + '/handwriting-model/tf/handwritingBCI/'

        if not os.path.isdir(outputDir):
            os.mkdir(outputDir)

        if not os.path.isdir(bashDir):
            os.mkdir(bashDir)

        args = {}
#         args['nSentences'] = 256
        args['nSentences'] = 100
        args['nSteps'] = sentenceLength
        args['binSize'] = 2
        # from https://github.com/first20hours/google-10000-english
        args['wordListFile'] = repoDir+'wordList/google-10000-english-usa.txt'
        args['rareWordFile'] = repoDir+'wordList/rareWordIdx.mat'
        args['snippetFile'] = rootDir+'RNNTrainingSteps/Step3_SyntheticSentences/' + \
            cvPart+'/'+dataDir+'_snippets.mat'
        args['accountForPenState'] = 1
        args['charDef'] = getHandwritingCharacterDefinitions()
        args['seed'] = datetime.datetime.now().microsecond

        argList = []
        for x in range(20):
            newArgs = args.copy()
            newArgs['saveFile'] = outputDir+'/bat_'+str(x)+'.tfrecord'
            newArgs['seed'] += x
            argList.append(newArgs)

        pool = multiprocessing.Pool(nParallelProcesses)
        results = pool.map(generateCharacterSequences, argList)

        pool.close()
        pool.join()

for n in range(len(results)):
    x = np.empty((1, 1200, 192))
    y = np.empty((1, 1200, 31))
    z = np.empty((1, 1200, 1))
    x = np.concatenate((x, np.array(results[n][0])))
    y = np.concatenate((y, np.array(results[n][1][:, :, 0:-1])))
    z = np.concatenate((z, np.array(results[n][1][:, :, -1, np.newaxis])))
    x = torch.from_numpy(x[1:, :, :])
    y = torch.from_numpy(y[1:, :, :])
    z = torch.from_numpy(z[1:, :, :])
    data = dict()
    data['inputs'] = x
    data['charLabels'] = y
    data['charStarts'] = z
    torch.save(data, baseDir + f'synthetic_sentences_{n}_torch.pt')
