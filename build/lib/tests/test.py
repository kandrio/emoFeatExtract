"""This is a test script for emoFeatExtract.py"""
import os
from pyAudioAnalysis import audioBasicIO
from emoFeatExtract2 import emoFeatExtract

from sklearn import svm

#gets label(emotion) that is mentioned in each sample name
def getEmotionLabel(x):
    return x[6:8]

#gets identity of the speaker, will be used to divide
#speakers into training and testing
def getSpeakerLabel(x):
    return x[18:20]

def trainortest(x):
    return x[15:17]

os.chdir('C:/Users/konst_000/Desktop/Σχολή/6ο Εξάμηνο/ΨΕΣ/Speech Emotion Recognition/Audio Database/Complete')
fileList = os.listdir('C:/Users/konst_000/Desktop/Σχολή/6ο Εξάμηνο/ΨΕΣ/Speech Emotion Recognition/Audio Database/Complete')
train_featureList = [] #list of numpy arrays used to store the extracted features of each training sample
train_labelList = []   #list of strings used to store the labels(emotions) for each training sample
test_featureList = []  #same for testing samples
test_labelList = []    #same for testing samples


for f in fileList:
    label = getEmotionLabel(f)
    if (label != '02' and label != '06'):
        continue
    [Fs, sample] = audioBasicIO.readAudioFile(f)
    sample = audioBasicIO.stereo2mono(sample) #feature extraction can be performed only on mono signals
    speaker = getSpeakerLabel(f)
    
    features = emoFeatExtract(sample, Fs, 0.050*Fs, 0.025*Fs)

    if(speaker == '05'):
        test_labelList.append(label)
        test_featureList.append(features)
    else:
        train_labelList.append(label)
        train_featureList.append(features)
    
clf = svm.SVC()
clf.fit(train_featureList, train_labelList)
predicted = clf.predict(test_featureList)

summ = 0
L = len(test_labelList)
for i in range(L):
    if test_labelList[i] == predicted[i]:
        summ = summ+1
summ = summ/L

print(test_labelList)
print(predicted)
print(summ)