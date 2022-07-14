import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from sklearn.utils import shuffle


def getName(filePath):
    return filePath.split('\\')[-1]


def importDataInfo(path):
    coloums = ['Center', 'Left', 'Right', 'Steering', 'Throttle', 'Brake', 'Speed']
    data = pd.read_csv(os.path.join(path, 'driving_log.csv'), names=coloums)
    data['Center'] = data['Center'].apply(getName)
    # print(data.head())
    print('Total Images Imported:', data.shape[0])
    return data


def balanceData(data, display=True):
    nBins = 31
    samplesPerBin = 1000
    hist, bins = np.histogram(data['Steering'], nBins)
    # print(bins)
    if display:
        center = (bins[:-1] + bins[1:]) * 0.5
        # print(center)
        plt.bar(center, hist, width=0.06)
        plt.plot((-1, 1), (samplesPerBin, samplesPerBin))
        plt.show()
    removeIndexList = []
    for j in range(nBins):
        binDataList = []
        for i in range(len(data['Steering'])):
            if data['Steering'][i] >= bins[j] and data['Steering'][i] <= bins[j + 1]:
                binDataList.append(i)
        binDataList = shuffle(binDataList)
        binDataList = binDataList[samplesPerBin:]
        removeIndexList.extend(binDataList)
    print('Removed Images:', len(removeIndexList))
    data.drop(data.index[removeIndexList], inplace=True)
    print('Remaining Images: ', len(data))
    if display:
        hist, _ = np.histogram(data['Steering'], nBins)
        plt.bar(center, hist, width=0.06)
        plt.plot((-1, 1), (samplesPerBin, samplesPerBin))
        plt.show()
    return data


def loadDate(path, data):
    imagePath = []
    steering = []
    for i in range(len(data)):
        indexedData = data.iloc[i]
        # print(indexedData)
        imagePath.append(os.path.join(path, 'IMG', indexedData[0]))
        steering.append(float(indexedData[3]))
    imagePath = np.asarray(imagePath)
    steering = np.asarray(steering)
    return imagePath, steering
