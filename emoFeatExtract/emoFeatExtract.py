import numpy as np
import math
from scipy.stats import iqr
from pyAudioAnalysis import audioFeatureExtraction as afe
from scipy import fft


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


def energyFrames(signalFrames):
    """Returns an array of energy values. Each
    value corresponds to the energy of a frame 
    in signal frames"""
    x = np.zeros(len(signalFrames))
    counter = 0
    for frame in signalFrames:
        x[counter] = frameEnergy(frame)
        counter = counter + 1
    return x


"""The following functions are an alternative solution to
pitch extraction using zero-crossing-rate. We track the pitch of a frame by finding 
the first peak of the autocorrelation function. Currently,
not as efficient as the zcr method. That's why they are 
commented out."""

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
#        return 0

   
def pitchFrames(signalFrames):
    """Returns an array of pitch(FO) values. Each
    value corresponds to a frame in signal frames.
    It uses the zero-crossing-rate method to extract
    the pitch of each frame. Used zero-crossing-rate function
    from pyAudioAnalysis Python Package"""
    x = np.zeros(len(signalFrames))
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

def durRisingSlopes(Frames, threshold = 0.0):
    """Computes durations (counted in number of frames)
    of RISING slopes.Returns an array of
    all durations of RISING slopes in the signal.
    
    ARGUMENTS:
        Frames: List of values of pitch or energy.
        threshold: Helping variable (belongs to [0,1] range)
                   that is used to ignore small falling slopes
                   between two more dominant rising slopes.
                   
    RETURNS:
        List of duration of all rising slopes."""
    counter, previous, durations = 1, Frames[0], []
    for i in range(len(Frames)):
        if Frames[i] > previous:
            counter = counter + 1
            peak = Frames[i]
        else:
            if counter > 1:
                if Frames[i] > (1-threshold)*peak:
                    counter = counter + 1
                else:
                    durations.append(counter)
                    counter = 1
        previous = Frames[i]
    if counter > 1:
        durations.append(counter)
    return durations


def durFallingSlopes(Frames, threshold = 0.0):
    """Computes durations (counted in number of frames)
    of FALLING slopes.Returns an array of
    all durations of FALLING slopes in the signal.
    
    ARGUMENTS:
        Frames: List of values of pitch or energy.
        threshold: Helping variable (belongs to [0,1] range)
                   that is used to ignore small rising slopes
                   between two more dominant falling slopes.
                   
    RETURNS:
        List of duration of all falling slopes."""
    counter, previous, durations = 1, Frames[0], []
    for i in range(len(Frames)):
        if Frames[i] < previous:
            counter = counter + 1
            peak = Frames[i]
        else:
            if counter > 1:
                if Frames[i] < (1+threshold)*peak:
                    counter = counter + 1
                else:
                    durations.append(counter)
                    counter = 1
        previous = Frames[i]
    if counter > 1:
        durations.append(counter)
    return durations
    
"""-------------------------------------------------------"""

"""Two functions below that will be used for the computation
of max, mean and median VALUE of rising/falling slopes
of pitch/energy"""

"""-------------------------------------------------------"""

def valRisingSlopes(Frames):
    """Computes values of RISING slopes.Returns a list of
    all values of RISING slopes in the signal."""
    previous, slopes = Frames[0], []
    for i in range(len(Frames)):
        if Frames[i] > previous:
            slopes.append(Frames[i] - previous)
        previous = Frames[i]
    return slopes
    
    
def valFallingSlopes(Frames):
    """Computes values of FALLING slopes.Returns a list of
    all values of FALLING slopes in the signal."""
    previous, slopes = Frames[0], []
    for i in range(len(Frames)):
        if Frames[i] < previous:
            slopes.append(previous - Frames[i])
        previous = Frames[i]
    return slopes

"""--------------------------------------------------------"""


def detect_leading_silence(signal, threshold = 20.0, step = 100):
    """This function removes preceding silence parts
    of the audio signal.It actually returns the index
    at which the speaker starts speaking.
    
    ARGUMENTS:
        signal: The input signal
        threshold: A dB threshold below which the corresponding
            signal frame is ignored (len(frame) == step).
        step: The size of signal frame of which the dB value is
        computed in each iteration.
    
    RETURNS:
        The index at which the speaker starts speaking."""
    trim_index = 0           # starting value of index to be returned
    assert step > 0
    while trim_index+step < len(signal) and (np.mean(abs(signal[trim_index:trim_index+step])) == 0 or 20*(math.log10(np.mean(abs(signal[trim_index:trim_index+step])))) < threshold):
        trim_index += step
    return trim_index


def mfcc(Frames, fs, mfccNumber):
    """This function uses "audioFeatureExtraction" from the
    pyAudioAnalysis package in order to compute the 
    Mel-Frequency Cepstral Coefficients of the signal.
    
    ARGUMENTS:
        Frames: List of signal frames.
        fs: Sampling frequency of the signal.
        mfccNumber: Number of MFC coefficients to be
            computed.
        
    RETURNS:
        The mean, maximum and minimum values of each 
        MFC coefficient."""        
    win = len(Frames[0])
    nFFT = int(win/2)
    [fbank, freqs] = afe.mfccInitFilterBanks(fs, nFFT)
    seq = []
    for frame in Frames:
        ft = abs(fft(frame))              
        ft = ft[0:nFFT]                   
        ft = ft/win
        seq.append(afe.stMFCC(ft, fbank, mfccNumber))
    Mean = np.mean(seq,0)
    Max = np.amax(seq,0)
    Min = np.amin(seq,0)
    return(np.concatenate((Mean,Max,Min)))
    

def spectralFlux(Frames):
    """
    Uses the Spectral Flux formula from "https://www.sciencedirect.com/topics/
    engineering/spectral-flux" to compute short time spectral flux values 
    for the signal.
    
    ARGUMENTS:
        Frames: List of signal frames
        
    RETURNS:
        List of values of spectral flux (seq) between successive frames.
    """
    start = 1
    seq = []
    for frame in Frames:
        current = abs(fft(frame))
        if start == 1:
            previous = current
            start = 0
            continue
        currsum = np.sum(current)
        prevsum = np.sum(previous)
        stFlux = np.sum((current / currsum - previous / prevsum) ** 2)
        seq.append(stFlux)
        previous = current
    return seq
                  


def emoFeatExtract(input_signal, fs, window, step):

    error_thres = 0.00
    mfccNumber = 12                                      #Number of Mel-Frequency Cepstral Coefficients
    
    window = int(window)
    step = int(step)
    signal = np.double(input_signal)
    vector1 = np.zeros(19)                               #Energy Features
    vector2 = np.zeros(19)                               #Pitch Features
    vector3 = np.zeros(3*mfccNumber+3)                   #Spectral Features
    
    #silence removal
    start_trim = detect_leading_silence(signal)
    end_trim = detect_leading_silence(np.flip(signal))
    duration = len(signal)    
    signal = signal[start_trim:duration-end_trim]        #Signal after silence removal 
    
    #signal normalization
    signal = signal / (2.0 ** 15)
    DC = signal.mean()
    MAX = (np.abs(signal)).max()
    signal = (signal - DC) / (MAX + 0.0000000001)
    signal = signal - DC   
    
    signalFrames = signalToFrames(signal, window, step)  #Break signal into frames
    EnergyValues = energyFrames(signalFrames)            #Compute energy value of each fram
    PitchValues = pitchFrames(signalFrames)              #Compute pitch value of each frame
    
    vector1[0] = max(EnergyValues)
    vector1[1] = np.mean(EnergyValues)
    vector1[2] = np.var(EnergyValues)
    
    drse = durRisingSlopes(EnergyValues, error_thres)
    dfse = durFallingSlopes(EnergyValues, error_thres)
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
    
    drsp = durRisingSlopes(PitchValues, error_thres)
    dfsp = durFallingSlopes(PitchValues, error_thres)
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
    
    vector3[0:3*mfccNumber] = mfcc(signalFrames, fs, mfccNumber)     #Computation of MFCC
    
    flux = spectralFlux(signalFrames)                                #Computation of Spectral Flux
    
    vector3[3*mfccNumber] = np.mean(flux)
    vector3[3*mfccNumber+1] = max(flux)
    vector3[3*mfccNumber+2] = min(flux)
  
    return np.concatenate((vector1,vector2,vector3))
