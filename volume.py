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



def HarmonicPower(wave,fsHz,fHz):
    #wave is digital sound vector
    #fsHz is sampling frequency
    #fHz is freq component we want to extract, e.g. 1000, 2000, 3000
    #power is mean square of speicifed frequency component, independent of phase
    #fHz = 1000
    t = np.arange(len(wave))
    t = t/fsHz
    sinVector = np.sin(2*np.pi*t*fHz)
    cosVector = np.cos(2*np.pi*t*fHz)
    A = np.mean(np.multiply(sinVector,wave))
    B = np.mean(np.multiply(cosVector,wave))
    power = 2*(A**2 + B**2)
    outDBSPL1000 = 10 * np.log10(power) + lCalib 
    vectorDb = 10 * np.log10(np.mean(np.square(sinVector))) #power of the digital wave, inDB
    return outDBSPL1000, power, outDBSPL1000 - vectorDb

def THD(wave, fsHz):
    p = []
    for i in range (1,7):
        _, power, _ = HarmonicPower(wave, fsHz, 1000*i)
        p.append(power)
    distortionPower = sum(p[1:])
    thd = math.sqrt(distortionPower/p[0])
    rms = math.sqrt(p[0])
    return thd, rms

def loudSpeakerCompressorDb(inDb,T,R,W): 
    WFinal = W if W >= 0 else 0
    if (inDb > (T+WFinal/2)):
        outDb = T + (inDb -T) /R
    elif (inDb > T-WFinal/2):
        outDb=inDb+(1/R-1)*(inDb-(T-WFinal/2))**2/(2*WFinal)
    else:
        outDb=inDb

    return outDb

def microphoneCompressorDb(outDbSpl,T,R,W): #pass correct gain, T, R, W are distinct from loudSpeaker's T, R, and W
    WFinal = W if W >= 0 else 0
    if (outDbSpl > (T+WFinal/2)):
        outDb = T + (outDbSpl -T) /R
    elif (outDbSpl > T-WFinal/2):
        outDb=outDbSpl+(1/R-1)*(outDbSpl-(T-WFinal/2))**2/(2*WFinal)
    else:
        outDb=outDbSpl

    return outDb
    

def CompressorDb(inDb,T,R,W): #microphone compressor, rename CompressorDb => microphoneCompressorDb, accept S but convert S to R
    WFinal = W if W >= 0 else 0
    if (inDb > (T+WFinal/2)):
        outDb = T + (inDb -T) /R
    elif (inDb > T-WFinal/2):
        outDb=inDb+(1/R-1)*(inDb-(T-WFinal/2))**2/(2*WFinal)
    else:
        outDb=inDb

    return outDb

def CalculateRMSError(inDBValues,outDBSPLValues,backgroundDBSPL,gainDBSPL,T,R,W,componentGainDBSPL):
    err = []
    for i in range(0,len(inDBValues)):
        err.append((outDBSPLValues[i] - SoundLevelModel(inDBValues[i],backgroundDBSPL,gainDBSPL,T,R,W,componentGainDBSPL))**2)
    rmsErrorDBSPL=np.sqrt(np.mean(err))
    return rmsErrorDBSPL

def CompressorInverseDb(outDb,T,R,W): #accept S but convert S to R

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

def SoundLevelModel(inDb,backgroundDbSpl,gainDbSpl,T,R,W,componentGainDBSPL): #include parameter for component gainDbSpl
    #currently does not include loudspeaker compression, enhance to 1) apply loudspeaker compression. there will be 2 gains: gain at short distance
    #and gain at long distance. make a note on physical data if collected near or far
    
    #0) recording needs to be labeled by near or far 
    #1) out_power = 10**((CompressorDb(inDb, T_speaker, R_speaker, W_speaker) + gain[i_distance])/10) + 10**(backgroundDbSpl/10)
    #2) outDbSpl = 10*math.log10(out_power) #done loudspeaker and background sound
    #3) outDbSpl = CompressorDb(outDbSpl, T_mic, R_mic, W_mic) #define S as S=1/R 
  
    outDbSpl=10*math.log10(10**(backgroundDbSpl/10)+10**((inDb+(gainDbSpl - componentGainDBSPL))/10))
    #outDbSpl=10*math.log10(10**(backgroundDbSpl/10)+10**((inDb+gainDbSpl)/10)) #adding gain and background noise
    outDbSpl = CompressorDb(outDbSpl, T, R, W)
    return outDbSpl

def SoundLevelCost(x,inDB,outDBSPL,componentGainDBSPL): #include parameter for component gainDbSpl
    backgroundDbSpl=x[0]
    gainDbSpl=x[1]
    T=x[2]
    R=x[3]
    W=x[4]
    cost=0
    for i in range(len(inDB)):
        cost=cost + (outDBSPL[i] - SoundLevelModel(inDB[i],backgroundDbSpl,gainDbSpl,T,R,W,componentGainDBSPL))**2
        #cost=cost + (outDBSPL[i] - SoundLevelModel(inDB[i],backgroundDbSpl,gainDbSpl,T,R,W))**2

    if W<0:
        cost = cost + 10*len(inDB)*W**2
    if W>20:
        cost = cost + 10*len(inDB)*(W-20)**2

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
    P = np.mean(np.square(recordedSineTone)) #this is where power is, keep old way to compare
    L = 10 * np.log10(P) + lCalib  # Sound level in dBSPL = outDBSPL
    vectorDb = 10 * np.log10(np.mean(np.square(sinewave))) #power of the digital wave, inDB
    return L - vectorDb, P, L, vectorDb

def run_volume_task(recordedSignalJson, sampleRate):
    sig = np.array(recordedSignalJson, dtype=np.float32)
    sinewave = generateSineWave(sampleRate) # Generate sine wave for comparison
    soundGainDbSPL, P, L, vectorDb = getCalibration(sig, sinewave)

     
    return soundGainDbSPL, P, L, vectorDb

def get_model_parameters(inDB,outDBSPL,lCalibFromPeer,componentGainDBSPL):
    global lCalib
    lCalib = lCalibFromPeer
    maxMeasuredDBSPL = np.max(outDBSPL)
    summed_gain = 0
    gain_count = 0
    for i in range(0,len(inDB)):
        if inDB[i] <= -20:
            summed_gain = summed_gain + (outDBSPL[i] - inDB[i])
            gain_count = gain_count + 1
    guess_gain = summed_gain/gain_count
    guesses=[70,guess_gain,0,100,20]
    guesses=scipy.optimize.fmin(SoundLevelCost,guesses,args=(inDB,outDBSPL,componentGainDBSPL))
    rmsError = CalculateRMSError(inDB,outDBSPL,guesses[0],guesses[1],guesses[2],guesses[3],guesses[4],componentGainDBSPL)
    return guesses[0], guesses[1], guesses[2], guesses[3], guesses[4], rmsError #backgroundDBSPL,gainDBSPL,T,R,W,rmsError

def run_volume_task_nonlinear(recordedSignalJson, sampleRate,lCalibFromPeer):
    global lCalib
    lCalib = 0
    sig = np.array(recordedSignalJson, dtype=np.float32)
    sinewave = generateSineWave(sampleRate) # Generate sine wave for comparison
    soundGainDbSPL, P, L, vectorDb = getCalibration(sig, sinewave)
    outDBSPL1000, P1000, soundGainDbSPL1000 = HarmonicPower(sig,sampleRate,1000)
    thd, rms = THD(sig,sampleRate)
    return soundGainDbSPL, P, L, vectorDb, outDBSPL1000, P1000, thd, rms, soundGainDbSPL1000
