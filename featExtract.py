'''
This Python piece of code implements feature extraction for every single
sample in RAVDESS audio database. Then, a panda DataFrame is created for
the extracted features as well as the labels (emotions) of each sample.
'''

import os
from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import audioFeatureExtraction
import pandas
import numpy as np
#import matplotlib.pyplot as plt


os.chdir("C:/Users/konst_000/Desktop/Σχολή/6ο Εξάμηνο/ΨΕΣ/Speech Emotion Recognition/Audio Database/Complete")


#gets label(emotion) that is mentioned in each sample name
def getEmotionLabel(x):
    return x[6:8]

fileList = os.listdir('C:/Users/konst_000/Desktop/Σχολή/6ο Εξάμηνο/ΨΕΣ/Speech Emotion Recognition/Audio Database/Complete');
featureList = [] #list of numpy arrays used to store the extracted features of each sample
labelList = [] #list of strings used to store the labels(emotions) for each sample


for f in fileList:
    [Fs, sample] = audioBasicIO.readAudioFile(f)
    sample = audioBasicIO.stereo2mono(sample) #feature extraction can be performed only on mono signals
    
    label = getEmotionLabel(f)
    labelList.append(label)

    features = audioFeatureExtraction.stFeatureExtraction(sample, Fs, 0.050*Fs, 0.025*Fs)
    featureList.append(features)
   
df1 = DataFrame(featureList)
df2 = DataFrame(labelList)
xl1 = df1.to_excel(r'C:/Users/konst_000/Desktop/Σχολή/6ο Εξάμηνο/ΨΕΣ/Speech Emotion Recognition/Audio Database/Features Database/featureList.xlsx')
xl2 = df2.to_excel(r'C:/Users/konst_000/Desktop/Σχολή/6ο Εξάμηνο/ΨΕΣ/Speech Emotion Recognition/Audio Database/Features Database/labelList.xlsx')
