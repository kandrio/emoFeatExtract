"""This is a test script for emoFeatExtract.py"""
"""In this script, after the feature extraction,
the K-folds cross-validation
technique is used, where K == 24 is the
number of different speakers on the RAVDESS
database. Currently, all samples that are used
contain calm (02) or fearful (06) emotions.
The classification takes place between these 2
emotional states."""

#current score: 
#[0.5    0.625  0.6875 0.625  0.9375 0.5    0.625  0.5625 0.5    0.625
# 0.5    0.4375 0.4375 0.6875 0.6875 0.4375 0.375  0.5625 0.5    0.5625
# 0.625  0.5625 0.6875 0.5625]
#0.5755208333333334

import os
from pyAudioAnalysis import audioBasicIO
from emoFeatExtract2 import emoFeatExtract
import numpy as np
from sklearn import svm
from sklearn.model_selection import cross_val_score as cvs

#gets label(emotion) that is mentioned in each sample name
def getEmotionLabel(x):
    return x[6:8]

#gets identity of the speaker
def getSpeakerLabel(x):
    return x[18:20]

#not used at the moment
def trainortest(x):
    return x[15:17]

def getThird(val):
    return val[2]

os.chdir('C:/Users/konst_000/Desktop/Σχολή/6ο Εξάμηνο/ΨΕΣ/Speech Emotion Recognition/Audio Database/Complete')
fileList = os.listdir('C:/Users/konst_000/Desktop/Σχολή/6ο Εξάμηνο/ΨΕΣ/Speech Emotion Recognition/Audio Database/Complete')
featureList = [] #list of lists used to store the extracted features of each training sample
labelList = []   #list of strings used to store the labels(emotions) for each training sample
speakerList = [] #list of strings used to store the speaker identity

for f in fileList:
    label = getEmotionLabel(f)
    if (label != '02' and label != '06'):
        continue
    [Fs, sample] = audioBasicIO.readAudioFile(f)
    sample = audioBasicIO.stereo2mono(sample) #feature extraction can be performed only on mono signals
    speaker = getSpeakerLabel(f)
    features = emoFeatExtract(sample, Fs, 0.050*Fs, 0.025*Fs)
    featureList.append(features)
    labelList.append(label)
    speakerList.append(speaker)

final = []

for i in range(len(featureList)):
    l = [featureList[i]]
    l.append(labelList[i])
    l.append(speakerList[i])
    final.append(l)

final.sort(key = getThird)

featureList = [] #list of lists used to store the extracted features of each training sample
labelList = []   #list of strings used to store the labels(emotions) for each training sample

for i in range(len(final)):
    featureList.append(final[i][0])
    labelList.append(final[i][1])
    
clf = svm.SVC(gamma = 'auto')

predictions = cvs(clf, featureList, labelList, cv = 24)

print(predictions)
print(np.mean(predictions))