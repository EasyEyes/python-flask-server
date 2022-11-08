import numpy as np
import math
import scipy.optimize #add to requirements

debug = False
ref_power = 2 * 10**-10  # power of the smallest sound you can hear
ref_pressure = 2 * 10**-5  # pressure of the smallest sound you can hear

# Recorded Signal Information
recordedSineTone = []
targetRange = [3.5, 4.5]  # [min, max] seconds
lCalib = 104.92978421490648 # for max volume with gain of .04


def CompressorDb(inDb,T,R,W):
    if (inDb > (T+W/2)):
        outDb = T + (inDb -T) /R
    elif (inDb > T-W/2):
        outDb=inDb+(1/R-1)*(inDb-(T-W/2))**2/(2*W)
    else:
        outDb=inDb

    return outDb

def CompressorInverseDb(outDb,T,R,W):

    if (outDb > (T+(W/2)/R)):
        inDb=T+R*(outDb-T)
    elif outDb>(T-W/2):
        a=1
        b=2*(W/(1/R-1)-(T-W/2))
        c=-outDb*2*W/(1/R-1)+(T-W/2)**2
        inDb2= -b/2 - math.sqrt(b^2-4*c)/2
        inDb = inDb2
    else:
        inDb=outDb

    return inDb

def SoundLevelModel(inDb,backgroundDbSpl,gainDbSpl,T,R,W):
    isolatedOutDbSpl=CompressorDb(inDb,T,R,W)+gainDbSpl
    outDbSpl=10*math.log10(10**(backgroundDbSpl/10)+10**(isolatedOutDbSpl/10))
    return outDbSpl

def SoundLevelCost(x,inDB,outDBSPL):
    backgroundDbSpl=x[0]
    gainDbSpl=x[1]
    T=x[2]
    R=x[3]
    W=x[4]
    cost=0
    for i in range(len(inDB)):
        cost=cost + (outDBSPL[i] - SoundLevelModel(inDB[i],backgroundDbSpl,gainDbSpl,T,R,W))**2

    return cost

def generateSineWave(sampleRate):
    start_time = 0
    end_time = 1
    time = np.arange(start_time, end_time, 1/sampleRate)
    theta = 0
    frequency = 1000  # Hz
    amplitude = 1
    return amplitude * \
        np.sin(2 * np.pi * frequency * time + theta)


def computeLCalib():
    P = np.mean(np.square(recordedSineTone))
    lCalib = 79 - 10 * np.log10(P)

def getCalibration(recordedSineTone, sinewave):
    # Power of the recorded signal
    P = np.mean(np.square(recordedSineTone))
    L = 10 * np.log10(P) + lCalib  # Sound level in dBSPL = outDBSPL
    vectorDb = 10 * np.log10(np.mean(np.square(sinewave))) #power of the digital wave, inDB
    return L - vectorDb, P, L, vectorDb

def run_volume_task(recordedSignalJson, sampleRate):
    sig = np.array(recordedSignalJson, dtype=np.float32)
    sinewave = generateSineWave(sampleRate) # Generate sine wave for comparison
    soundGainDbSPL, P, L, vectorDb = getCalibration(sig, sinewave)
     
    return soundGainDbSPL, P, L, vectorDb

def get_model_parameters(inDB,outDBSPL):
    guesses=[70,130,-25,10,40]
    guesses=scipy.optimize.fmin(SoundLevelCost,guesses,args=(inDB,outDBSPL))
    return guesses[0], guesses[1], guesses[2], guesses[3], guesses[4]

def run_volume_task_nonlinear(recordedSignalJson, sampleRate):
    sig = np.array(recordedSignalJson, dtype=np.float32)
    sinewave = generateSineWave(sampleRate) # Generate sine wave for comparison
    soundGainDbSPL, P, L, vectorDb = getCalibration(sig, sinewave)
    
    return soundGainDbSPL, P, L, vectorDb
