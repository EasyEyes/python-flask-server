import numpy as np

debug = False
ref_power = 2 * 10**-10  # power of the smallest sound you can hear
ref_pressure = 2 * 10**-5  # pressure of the smallest sound you can hear

# Recorded Signal Information
recordedSineTone = []
targetRange = [3.5, 4.5]  # [min, max] seconds
lCalib = 104.92978421490648 # for max volume with gain of .04


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
    print(lCalib)


def getCalibration(recordedSineTone, sinewave):
    # Power of the recorded signal
    P = np.mean(np.square(recordedSineTone))
    L = 10 * np.log10(P) + lCalib  # Sound level in dB
    vectorDb = 10 * np.log10(np.mean(np.square(sinewave)))
    return L - vectorDb, P, L, vectorDb

def run_volume_task(recordedSignalJson, sampleRate):
    sig = np.array(recordedSignalJson, dtype=np.float32)
    sinewave = generateSineWave(sampleRate) # Generate sine wave for comparison
    soundGainDbSPL, P, L, vectorDb = getCalibration(sig, sinewave)
    
    return soundGainDbSPL, P, L, vectorDb