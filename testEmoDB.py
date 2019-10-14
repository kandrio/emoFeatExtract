#Emotional Predictions
#[0.5806451612903226, 0.675, 0.6206896551724138, 0.4, 0.8235294117647058, 0.4117647058823529, 0.5, 0.5853658536585366, 0.6285714285714286, 0.58]

#0.5805566216339761

#Binary Activation
#[0.9183673469387755, 0.9482758620689655, 0.8372093023255814, 0.8157894736842105, 0.8, 0.8571428571428571, 0.9344262295081968, 0.927536231884058, 0.9464285714285714, 0.8028169014084507]

#0.8787992776389666

#Binary Valence
#[0.8974358974358975, 0.8809523809523809, 0.6333333333333333, 0.8571428571428571, 0.7714285714285715, 0.6363636363636364, 0.5833333333333334, 0.8048780487804879, 0.6764705882352942, 0.717948717948718]

#0.745928736495451

import os
from pyAudioAnalysis import audioBasicIO
from emoFeatExtract import emoFeatExtract
import numpy as np
from sklearn import svm

#gets label(emotion) that is mentioned in each sample name
def getEmotionLabel(x):
    return x[5]

#gets identity of the speaker
def getSpeakerLabel(x):
    return x[0:2]

os.chdir('C:/Users/konst_000/Desktop/Σχολή/6ο Εξάμηνο/ΨΕΣ/Speech Emotion Recognition/Audio Database/Complete Berlin')
fileList = os.listdir('C:/Users/konst_000/Desktop/Σχολή/6ο Εξάμηνο/ΨΕΣ/Speech Emotion Recognition/Audio Database/Complete Berlin')
featureList = []
featureListAct = [] #list of lists used to store the extracted features of each training sample
featureListVal = []
labelList = []      #list of strings used to store the labels(emotions) for each training sample
labelListAct = []   #list of strings used to store the labels(emotions) for each training sample
labelListVal = []
speakerList = []
speakerListAct = [] #list of strings used to store the speaker identity
speakerListVal = []

for f in fileList:
    [Fs, sample] = audioBasicIO.readAudioFile(f)
    sample = audioBasicIO.stereo2mono(sample) #feature extraction can be performed only on mono signals
    features = emoFeatExtract(sample, Fs, 0.050*Fs, 0.025*Fs)
    label = getEmotionLabel(f)
    speaker = getSpeakerLabel(f)
    if(label == 'L' or label == 'E' or label == 'F' or label == 'T' or label == 'N'):
        featureList.append(features)
        labelList.append(label)
        speakerList.append(speaker)
    #Binary Activation Labels
    if (label == 'L' or label == 'E' or label == 'N' or label == 'T'):
        labelListAct.append('Low')
    else:
        labelListAct.append('High')
    featureListAct.append(features)
    speakerListAct.append(speaker)
    if (label == 'W' or label == 'T'):
        labelListVal.append('Negative')
        featureListVal.append(features)
        speakerListVal.append(speaker)
    else: 
        if (label == 'F' or label == 'N'):
            labelListVal.append('Positive')
            featureListVal.append(features)
            speakerListVal.append(speaker)
    

final = []
finalAct = []
finalVal = []

for i in range(len(featureList)):
    l = [featureList[i]]
    l.append(labelList[i])
    l.append(speakerList[i])
    final.append(l)
    
for i in range(len(featureListAct)):
    l = [featureListAct[i]]
    l.append(labelListAct[i])
    l.append(speakerListAct[i])
    finalAct.append(l)
    
for i in range(len(featureListVal)):
    l = [featureListVal[i]]
    l.append(labelListVal[i])
    l.append(speakerListVal[i])
    finalVal.append(l)
    
os.chdir('C:/Users/konst_000/Desktop/Σχολή/6ο Εξάμηνο/ΨΕΣ/Speech Emotion Recognition/Audio Database/FeaturesDB/BERLIN')
    
np.save('featureList.npy', featureList)
np.save('labelList.npy', labelList)
np.save('featureListAct.npy', featureListAct)
np.save('labelListAct.npy', labelListAct)
np.save('featureListVal.npy', featureListVal)
np.save('labelListVal.npy', labelListVal)
#
clf = svm.SVC(gamma = 'auto')

os.chdir('C:/Users/konst_000/Desktop/Σχολή/6ο Εξάμηνο/ΨΕΣ/Speech Emotion Recognition/Audio Database/FeaturesDB/BERLIN')

featureList = np.load('featureList.npy')
labelList = np.load('labelList.npy')   #list of strings used to store the labels(emotions) for each training sample
featureListAct = np.load('featureListAct.npy')
labelListAct = np.load('labelListAct.npy')   #list of strings used to store the labels(emotions) for each training sample
featureListVal = np.load('featureListVal.npy')
labelListVal = np.load('labelListVal.npy')

differentSpeakers = ['03','08','09','10','11','12','13','14','15','16']

out = []
outAct = []
outVal = []

for s in differentSpeakers:
    trainFeatureList = []
    trainLabelList = []
    trainFeatureListAct = []
    trainLabelListAct = []
    trainFeatureListVal = []   
    trainLabelListVal = []
    testFeatureList = []
    testLabelList = []
    testFeatureListAct = []
    testLabelListAct = []
    testFeatureListVal = []
    testLabelListVal = []
    for instance in final:
        if instance[2] == s:
            testFeatureList.append(instance[0])
            testLabelList.append(instance[1])
        else:
            trainFeatureList.append(instance[0])
            trainLabelList.append(instance[1])
    clf.fit(trainFeatureList, trainLabelList)
    out.append(clf.score(testFeatureList,testLabelList))
    for instance in finalAct:
        if instance[2] == s:
            testFeatureListAct.append(instance[0])
            testLabelListAct.append(instance[1])
        else:
            trainFeatureListAct.append(instance[0])
            trainLabelListAct.append(instance[1])
    clf.fit(trainFeatureListAct, trainLabelListAct)
    outAct.append(clf.score(testFeatureListAct, testLabelListAct))
    for instance in finalVal:
        if instance[2] == s:
            testFeatureListVal.append(instance[0])
            testLabelListVal.append(instance[1])
        else:
            trainFeatureListVal.append(instance[0])
            trainLabelListVal.append(instance[1])
    clf.fit(trainFeatureListVal, trainLabelListVal)
    outVal.append(clf.score(testFeatureListVal,testLabelListVal))
    
print('Emotional Predictions')
print(out)
print(np.mean(out))
print('Binary Activation')
print(outAct)
print(np.mean(outAct))
print('Binary Valence')
print(outVal)
print(np.mean(outVal))