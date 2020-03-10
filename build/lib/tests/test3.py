"""This is a test script for emoFeatExtract.py"""
"""In this script, after the feature extraction,
the K-folds cross-validation
technique is used, where K == 24 is the
number of different speakers on the RAVDESS
database. Currently we classificate the samples on binary Activation
and binary Valence."""

#Current Score

#Binary Activation
#[0.71666667 0.76666667 0.76666667 0.68333333 0.71666667 0.78333333
# 0.73333333 0.81666667 0.56666667 0.73333333 0.7        0.71666667
# 0.73333333 0.78333333 0.71666667 0.73333333 0.75       0.65
# 0.56666667 0.7        0.68333333 0.71666667 0.78333333 0.8       ]
#0.7215277777777778

#Binary Valence
#[0.71666667 0.53333333 0.68333333 0.6        0.56666667 0.5
# 0.65       0.65       0.56666667 0.61666667 0.58333333 0.56666667
# 0.55       0.66666667 0.61666667 0.6        0.63333333 0.68333333
# 0.68333333 0.68333333 0.6        0.53333333 0.58333333 0.68333333]
#0.6145833333333334

import os
from pyAudioAnalysis import audioBasicIO
from emoFeatExtract import emoFeatExtract
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

def getFourth(val):
    return val[3]

os.chdir('C:/Users/konst_000/Desktop/Σχολή/6ο Εξάμηνο/ΨΕΣ/Speech Emotion Recognition/Audio Database/Complete')
fileList = os.listdir('C:/Users/konst_000/Desktop/Σχολή/6ο Εξάμηνο/ΨΕΣ/Speech Emotion Recognition/Audio Database/Complete')
featureList = [] #list of lists used to store the extracted features of each training sample
labelListAct = []   #list of strings used to store the labels(emotions) for each training sample
labelListVal = []
speakerList = [] #list of strings used to store the speaker identity

for f in fileList:
    label = getEmotionLabel(f)
    [Fs, sample] = audioBasicIO.readAudioFile(f)
    sample = audioBasicIO.stereo2mono(sample) #feature extraction can be performed only on mono signals
    speaker = getSpeakerLabel(f)
    features = emoFeatExtract(sample, Fs, 0.050*Fs, 0.025*Fs)
    featureList.append(features)
    #Binary Activation Labels
    if (label == '01' or label == '02' or label == '04' or label == '07'):
        labelListAct.append('Low')
    else:
        labelListAct.append('High')   
    if (label == '04' or label == '05' or label == '06' or label == '07'):
        labelListVal.append('Negative')
    else:
        labelListVal.append('Positive')
    speakerList.append(speaker)

final = []

for i in range(len(featureList)):
    l = [featureList[i]]
    l.append(labelListAct[i])
    l.append(labelListVal[i])
    l.append(speakerList[i])
    final.append(l)

final.sort(key = getFourth)

featureList = [] #list of lists used to store the extracted features of each training sample
labelListAct = []   #list of strings used to store the labels(emotions) for each training sample
labelListVal = []

for i in range(len(final)):
    featureList.append(final[i][0])
    labelListAct.append(final[i][1])
    labelListVal.append(final[i][2])
    
clf = svm.SVC(gamma = 'auto')

predictionsAct = cvs(clf, featureList, labelListAct, cv = 24)
predictionsVal = cvs(clf, featureList, labelListVal, cv = 24)

print('Binary Activation')
print(predictionsAct)
print(np.mean(predictionsAct))
print('Binary Valence')
print(predictionsVal)
print(np.mean(predictionsVal))