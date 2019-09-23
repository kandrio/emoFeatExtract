import numpy as np
from scipy.stats import iqr
from scipy.signal import butter, filtfilt, find_peaks
from pyAudioAnalysis import audioFeatureExtraction as afe

def signalToFrames(signal, window, step):
    """Divides given signal in frames of length of 'window' using
    a 'step'. Returns numpy array of frames."""
    length, curr_start, frames = np.size(signal), 0, []    
    while(curr_start + window < length):
        x = signal[curr_start:curr_start + window]
        frames.append(x)
        curr_start = curr_start + step
    return frames


def frameEnergy(frame):
    """Computes energy of frame signal"""
    return np.sum(frame**2) / np.float64(np.size(frame))   
#np.float64 may have to change
#depends on audio reading algorithm

def energyFrames(signalFrames):
    """Returns an array of energy values. Each
    value corresponds to the energy of a frame 
    in signal frames"""
    x = np.zeros(np.size(signalFrames))
    counter = 0
    for frame in signalFrames:
        x[counter] = frameEnergy(frame)
        counter = counter + 1
    return x
#np.float64 may have to change
#depends on audio reading algorithm
    

#def autocorrelation(x):
#    """Computes the autocorrelation of signal
#    frame x"""
#    result = np.correlate(x, x, mode = 'full')
#    return result[np.size(result)//2:]
#
#def framePitch(frame):
#    """Computes pitch (F0) of frame signal"""
#    x = autocorrelation(frame)
#    peaks, _ = find_peaks(x)
#    if np.size(peaks) >= 1:
#        return peaks[0]
#    else:
#        return 0 #lysi anagkhs

   
def pitchFrames(signalFrames):
    """Returns an array of pitch(FO) values. Each
    value corresponds to a frame in signal frames"""
    x = np.zeros(np.size(signalFrames))
    counter = 0
    for frame in signalFrames:
        #x[counter] = framePitch(frame)
        x[counter] = afe.stZCR(frame)
        counter = counter + 1
    return x



"""No additional functions for max, mean, variance of 
energy and pitch. Numpy functions do the job(max, np.mean, np.var)"""

"""Two functions below that will be used for the computation
of max, mean and median DURATION of rising/falling slopes
of pitch/energy"""

"""-------------------------------------------------------------"""

def durRisingSlopes(Frames):
    """Computes durations (counted in number of frames)
    of RISING slopes.Returns an array of
    all durations of RISING slopes in the signal."""
    counter, previous, durations = 0, 0, []
    for frame in Frames:
        if frame > previous:
            counter = counter + 1
        else:
            if counter > 1:
                durations.append(counter)
                counter = 1
        previous = frame
    if counter > 1:
        durations.append(counter)
    if len(durations) == 0:
        return 0
    else:
        return durations


def durFallingSlopes(Frames):
    """Computes durations (counted in number of frames)
    of FALLING slopes.Returns an array of
    all durations of FALLING slopes in the signal."""
    counter, previous, durations = 1, 0, []
    for frame in Frames:
        if frame < previous:
            counter = counter + 1
        else:
            if counter > 1:
                durations.append(counter)
                counter = 1
        previous = frame
    if counter > 1:
        durations.append(counter)
    if len(durations) == 0:
        return 0
    else:
        return durations
    
"""-------------------------------------------------------"""

"""Two functions below that will be used for the computation
of max, mean and median VALUE of rising/falling slopes
of pitch/energy"""

"""-------------------------------------------------------"""

def valRisingSlopes(Frames):
    """Computes values of RISING slopes.Returns an array of
    all values of RISING slopes in the signal."""
    previous, slopes, start = 0, [], 1
    for frame in Frames:
        if start == 1:
            previous = frame
            start = 0
            continue
        if frame > previous:
            slopes.append(frame - previous)
        previous = frame
    if len(slopes) == 0:
        return 0
    else:
        return slopes
    
def valFallingSlopes(Frames):
    """Computes values of FALLING slopes.Returns an array of
    all values of FALLING slopes in the signal."""
    previous, slopes, start = 0, [], 1
    for frame in Frames:
        if start == 1:
            previous = frame
            start = 0
            continue
        if frame < previous:
            slopes.append(previous - frame)
        previous = frame
    if len(slopes) == 0:
        return 0
    else:
        return slopes 

"""--------------------------------------------------------"""


def emoFeatExtract(input_signal, fs, window, step):
   
    #signal filtering
    nyq = 0.5*fs
    
    low = 75/nyq
    high = 500/nyq
    b, a = butter(5, [low,high], btype='band')
    signal = filtfilt(b, a, input_signal)
    
    window = int(window)
    step = int(step)
    signal = np.double(input_signal)
    vector1 = np.zeros(19)
    vector2 = np.zeros(19)

    #signal normalization
    signal = signal / (2.0 ** 15)
    DC = signal.mean()
    MAX = (np.abs(signal)).max()
    signal = (signal - DC) / (MAX + 0.0000000001)
    
    
    signalFrames = signalToFrames(signal, window, step)
    EnergyValues = energyFrames(signalFrames)
    PitchValues = pitchFrames(signalFrames)
    
    vector1[0] = max(EnergyValues)
    vector1[1] = np.mean(EnergyValues)
    vector1[2] = np.var(EnergyValues)
    
    drse = durRisingSlopes(EnergyValues)
    dfse = durFallingSlopes(EnergyValues)
    vrse = valRisingSlopes(EnergyValues)
    vfse = valFallingSlopes(EnergyValues)
    
    vector1[3] = max(drse)
    vector1[4] = np.mean(drse)
    vector1[5] = np.median(drse)
    vector1[6] = max(dfse)
    vector1[7] = np.mean(dfse)
    vector1[8] = np.median(dfse)
    vector1[9] = max(vrse)
    vector1[10] = np.mean(vrse)
    vector1[11] = np.median(vrse)
    vector1[12] = max(vfse)
    vector1[13] = np.mean(vfse)
    vector1[14] = np.median(vfse)
    vector1[15] = iqr(vrse)
    vector1[16] = iqr(vfse)
    vector1[17] = iqr(drse)
    vector1[18] = iqr(dfse)
    
    
    
    vector2[0] = max(PitchValues)
    vector2[1] = np.mean(PitchValues)
    vector2[2] = np.var(PitchValues)
    
    drsp = durRisingSlopes(PitchValues)
    dfsp = durFallingSlopes(PitchValues)
    vrsp = valRisingSlopes(PitchValues)
    vfsp = valFallingSlopes(PitchValues)
    
    vector2[3] = max(drsp)
    vector2[4] = np.mean(drsp)
    vector2[5] = np.median(drsp)
    vector2[6] = max(dfsp)
    vector2[7] = np.mean(dfsp)
    vector2[8] = np.median(dfsp)
    vector2[9] = max(vrsp)
    vector2[10] = np.mean(vrsp)
    vector2[11] = np.median(vrsp)
    vector2[12] = max(vfsp)
    vector2[13] = np.mean(vfsp)
    vector2[14] = np.median(vfsp)
    vector2[15] = iqr(vrsp)
    vector2[16] = iqr(vfsp)
    vector2[17] = iqr(drsp)
    vector2[18] = iqr(dfsp)
    
    return np.concatenate((vector1,vector2))