"""Extracting all Energy related features from the signal"""
import numpy as np
from scipy.stats import iqr

def frameEnergy(frame):
    """Computes energy of frame signal"""
    return np.sum(frame**2) / np.float64(len(frame))

def maxSignalEnergy(signalFrames):
    """Computes maximum frame energy of framed signal"""
    maxValue = 0
    for frame in signalFrames:
        x = frameEnergy(frame)
        if x < maxValue:
            maxValue = x
    return maxValue

def meanSignalEnergy(signalFrames):
    """Computes mean signal energy of framed signal"""
    sum = 0
    for frame in signalFrames:
        x = frameEnergy(frame)
        sum = x+sum 
    return sum / np.float64(len(signalFrames))

def varEnergy(signalFrames):
    """Computes variance of energy of framed signal"""
    x = np.zeros(np.float64(len(signalFrames)))
    counter =  0
    for frame in signalFrames:
        x[counter] = frameEnergy(frame)
        counter = counter + 1
    return np.var(x)

def maxDurRiEnSlo(signalFrames):
    """Computes maximum duration (in frames) of
    RISING slopes of energy"""
    counter = 0
    maximum = 0
    previous = 0
    current = 0
    for frame in signalFrames:
        current = frameEnergy(frame)
        if current > previous:
            previous = current
            counter = counter+1
            maximum = max(maximum,counter)
        else:
            counter = 1
            previous = current
    return maximum

def maxDurFaEnSlo(signalFrames):
    """Computes maximum duration (in frames) of
    FALLING slopes of energy"""
    counter = 0
    maximum = 0
    previous = 0
    current = 0
    for frame in signalFrames:
        current = frameEnergy(frame)
        if current < previous:
            previous = current
            counter = counter+1
            maximum = max(maximum,counter)
        else:
            counter = 1
            previous = current
    return maximum

def meanDurRiEnSlo(signalFrames):
    """Computes mean duration (in frames) of
    RISING slopes of energy"""
    counter, previous, current, rising, sum = 0, 0, 0, 0, 0
    for frame in signalFrames:
        current = frameEnergy(frame)
        if current > previous:
            counter = counter + 1
        else:
            if counter > 1:
                rising = rising + 1
                sum = sum + counter
                counter = 1
        previous = current
    if counter > 1:
        rising = rising + 1
        sum = sum + counter
    if sum == 0:
        return 0
    else:
        return (sum/rising)


def meanDurFaEnSlo(signalFrames):
    """Computes mean duration (in frames) of
    FALLING slopes of energy"""
    counter, previous, current, falling, sum = 1, 0, 0, 0, 0
    for frame in signalFrames:
        current = frameEnergy(frame)
        if current < previous:
            counter = counter + 1
        else:
            if counter > 1:
                falling = falling + 1
                sum = sum + counter
                counter = 1
        previous = current
    if counter > 1:
        falling = falling + 1
        sum = sum + counter
    if sum == 0:
        return 0
    else:
        return (sum/falling)
    
def medianDurRiEnSlo(signalFrames):
    """Computes median duration (in frames) of
    RISING slopes of energy"""
    counter, previous, current, durations = 0, 0, 0, np.array([])
    for frame in signalFrames:
        current = frameEnergy(frame)
        if current > previous:
            counter = counter + 1
        else:
            if counter > 1:
                durations = np.append(durations,counter)
                counter = 1
        previous = current
    if counter > 1:
        durations = np.append(durations,counter)
    if durations.size == 0:
        return 0
    else:
        return np.median(durations)
    

def medianDurFaEnSlo(signalFrames):
    """Computes median duration (in frames) of
    FALLING slopes of energy"""
    counter, previous, current, durations = 1, 0, 0, np.array([])
    for frame in signalFrames:
        current = frameEnergy(frame)
        if current < previous:
            counter = counter + 1
        else:
            if counter > 1:
                durations = np.append(durations,counter)
                counter = 1
        previous = current
    if counter > 1:
        durations = np.append(durations,counter)
    if durations.size == 0:
        return 0
    else:
        return np.median(durations)

    
def maxValRiEnSlo(signalFrames):
    """Computes maximum value of
    RISING slopes of energy"""
    previous, current, maximum, start = 0, 0, 0, 1
    for frame in signalFrames:
        current = frameEnergy(frame)
        if start == 1:
            previous = current
            start = 0
            continue
        candidate = current - previous
        if candidate > maximum:
            maximum = candidate
        else:
            previous = current
    return maximum
    

def maxValFaEnSlo(signalFrames):
    """Computes maximum value of
    FALLING slopes of energy"""
    previous, current, maximum, start = 0, 0, 0, 1
    for frame in signalFrames:
        current = frameEnergy(frame)
        if start == 1:
            previous = current
            start = 0
            continue
        candidate = previous - current
        if candidate > maximum:
            maximum = candidate
        else:
            previous = current
    return maximum
    

def meanValRiEnSlo(signalFrames):
    """Computes mean value of
    RISING slopes of energy"""
    previous, current, rising, sum, start = 0, 0, 0, 0, 1
    for frame in signalFrames:
        current = frameEnergy(frame)
        if start == 1:
            previous = current
            start = 0
            continue
        if current > previous:
            sum = sum + current - previous
            rising = rising + 1
        previous = current
    if sum == 0:
        return 0
    else:
        return (sum/rising)    
    

def meanValFaEnSlo(signalFrames):
    """Computes mean value of
    FALLING slopes of energy"""
    previous, current, falling, sum, start = 0, 0, 0, 0, 1
    for frame in signalFrames:
        current = frameEnergy(frame)
        if start == 1:
            previous = current
            start = 0
            continue
        if current < previous:
            sum = sum + previous - current
            falling = falling + 1
        previous = current
    if sum == 0:
        return 0
    else:
        return (sum/falling)
    

def medianValRiEnSlo(signalFrames):
    """Computes median value of
    RISING slopes of energy"""
    previous, current, slopes, start = 0, 0, np.array([]), 1
    for frame in signalFrames:
        current = frameEnergy(frame)
        if start == 1:
            previous = current
            start = 0
            continue
        if current > previous:
            slopes = np.append(slopes,current - previous)
        previous = current
    if slopes.size == 0:
        return 0
    else:
        return np.median(slopes)  
    
def medianValFaEnSlo(signalFrames):
    """Computes median value of
    FALLING slopes of energy"""
    previous, current, slopes, start = 0, 0, np.array([]), 1
    for frame in signalFrames:
        current = frameEnergy(frame)
        if start == 1:
            previous = current
            start = 0
            continue
        if current < previous:
            slopes = np.append(slopes,previous - current)
        previous = current
    if slopes.size == 0:
        return 0
    else:
        return np.median(slopes) 
    
    

