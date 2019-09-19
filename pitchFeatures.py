"""Extracting all Pitch related features from the signal"""
import numpy as np
#import parselmouth as pm
import pyAudioAnalysis as pAA

def autocorr(frame):
    result = np.correlate(frame, frame, mode='full')
    return result[result.size//2:]

def maxPitch(pitchFrames):
    return max(pitchFrames)

def meanPitch(pitchFrames):
    return np.mean(pitchFrames)

def varPitch(pitchFrames):
    return np.var(pitchFrames)

def maxDurRiPiSlo(pitchFrames):
    """Computes maximum duration (in frames) of
    RISING slopes of pitch"""
    counter, maximum, previous = 0, 0, 0
    for frame in pitchFrames:
        if frame > previous:
            previous = frame
            counter = counter+1
            maximum = max(maximum,counter)
        else:
            counter = 1
            previous = frame
    return maximum

def maxDurFaPiSlo(pitchFrames):
    """Computes maximum duration (in frames) of
    FALLING slopes of pitch"""
    counter, maximum, previous = 0, 0, 0
    for frame in pitchFrames:
        if frame < previous:
            previous = frame
            counter = counter+1
            maximum = max(maximum,counter)
        else:
            counter = 1
            previous = frame
    return maximum

def meanDurRiPiSlo(pitchFrames):
    """Computes mean duration (in frames) of
    RISING slopes of pitch"""
    counter, previous, rising, sum = 0, 0, 0, 0
    for frame in pitchFrames:
        if frame > previous:
            counter = counter + 1
        else:
            if counter > 1:
                rising = rising + 1
                sum = sum + counter
                counter = 1
        previous = frame
    if counter > 1:
        rising = rising + 1
        sum = sum + counter
    if sum == 0:
        return 0
    else:
        return (sum/rising)


def meanDurFaPiSlo(pitchFrames):
    """Computes mean duration (in frames) of
    FALLING slopes of pitch"""
    counter, previous, falling, sum = 1, 0, 0, 0
    for frame in pitchFrames:
        if frame < previous:
            counter = counter + 1
        else:
            if counter > 1:
                falling = falling + 1
                sum = sum + counter
                counter = 1
        previous = frame
    if counter > 1:
        falling = falling + 1
        sum = sum + counter
    if sum == 0:
        return 0
    else:
        return (sum/falling)
    
def medianDurRiPiSlo(pitchFrames):
    """Computes median duration (in frames) of
    RISING slopes of pitch"""
    counter, previous, durations = 0, 0, np.array([])
    for frame in pitchFrames:
        if frame > previous:
            counter = counter + 1
        else:
            if counter > 1:
                durations = np.append(durations,counter)
                counter = 1
        previous = frame
    if counter > 1:
        durations = np.append(durations,counter)
    if durations.size == 0:
        return 0
    else:
        return np.median(durations)
    

def medianDurFaPiSlo(pitchFrames):
    """Computes median duration (in frames) of
    FALLING slopes of pitch"""
    counter, previous, durations = 1, 0, np.array([])
    for frame in pitchFrames:
        if frame < previous:
            counter = counter + 1
        else:
            if counter > 1:
                durations = np.append(durations,counter)
                counter = 1
        previous = frame
    if counter > 1:
        durations = np.append(durations,counter)
    if durations.size == 0:
        return 0
    else:
        return np.median(durations)

    
def maxValRiPiSlo(pitchFrames):
    """Computes maximum value of
    RISING slopes of pitch"""
    previous, maximum, start = 0, 0, 1
    for frame in pitchFrames:
        if start == 1:
            previous = frame
            start = 0
            continue
        candidate = frame - previous
        if candidate > maximum:
            maximum = candidate
        else:
            previous = frame
    return maximum
    

def maxValFaPiSlo(pitchFrames):
    """Computes maximum value of
    FALLING slopes of pitch"""
    previous, maximum, start = 0, 0, 1
    for frame in pitchFrames:
        if start == 1:
            previous = frame
            start = 0
            continue
        candidate = previous - frame
        if candidate > maximum:
            maximum = candidate
        else:
            previous = frame
    return maximum
    

def meanValRiPiSlo(pitchFrames):
    """Computes mean value of
    RISING slopes of pitch"""
    previous, rising, sum, start = 0, 0, 0, 1
    for frame in pitchFrames:
        if start == 1:
            previous = frame
            start = 0
            continue
        if frame > previous:
            sum = sum + frame - previous
            rising = rising + 1
        previous = frame
    if sum == 0:
        return 0
    else:
        return (sum/rising)    
    

def meanValFaPiSlo(pitchFrames):
    """Computes mean value of
    FALLING slopes of pitch"""
    previous, falling, sum, start = 0, 0, 0, 1
    for frame in pitchFrames:
        if start == 1:
            previous = frame
            start = 0
            continue
        if frame < previous:
            sum = sum + previous - frame
            falling = falling + 1
        previous = frame
    if sum == 0:
        return 0
    else:
        return (sum/falling)
    

def medianValRiPiSlo(pitchFrames):
    """Computes median value of
    RISING slopes of pitch"""
    previous, slopes, start = 0, np.array([]), 1
    for frame in pitchFrames:
        if start == 1:
            previous = frame
            start = 0
            continue
        if frame > previous:
            slopes = np.append(slopes,frame - previous)
        previous = frame
    if slopes.size == 0:
        return 0
    else:
        return np.median(slopes)  
    
def medianValFaPiSlo(pitchFrames):
    """Computes median value of
    FALLING slopes of pitch"""
    previous, slopes, start = 0, np.array([]), 1
    for frame in pitchFrames:
        if start == 1:
            previous = frame
            start = 0
            continue
        if frame < previous:
            slopes = np.append(slopes,previous - frame)
        previous = frame
    if slopes.size == 0:
        return 0
    else:
        return np.median(slopes) 

