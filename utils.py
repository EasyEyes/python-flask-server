from cmath import inf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from impulse_response import run_ir_task, recover_signal, g_filter
# from inverted_impulse_response import run_iir_task
from scipy.signal import convolve
from scipy.fft import fft, rfft
from scipy.io import wavfile

parser = argparse.ArgumentParser(description="Ingest audio samples from client.")
parser.add_argument("--sampleRate", type=int, help="Sample rate of payload, in hz")
parser.add_argument("--payload", nargs="+", type=float, help="Array of audio samples")
parser.add_argument("--debug", action="store_true", help="Print debug info")

def plot_power_spectrum(sig, sampling_rate=96000):

    time = np.arange(0, len(sig), 1/sampling_rate)

    fourier_transform = fft(sig)

    abs_fourier_transform = np.abs(fourier_transform)

    power_spectrum = np.square(abs_fourier_transform)

    frequency = np.linspace(0, sampling_rate/2, len(power_spectrum))

    plt.plot(frequency, power_spectrum)
    plt.show()

def plot_frequency_spectrum(s, s_name, plot_zoom=False, xlim=[0, .001], dt=96000):
    max_time = (s.size/dt) 
    Fs = 1 / dt  # sampling frequency
    t = np.linspace(0, max_time, s.size) 
    
    fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(7, 7))

    # plot time signal:
    axs[0, 0].set_title(s_name)
    axs[0, 0].plot(t, s, color='C0')
    axs[0, 0].set_xlabel("Time [s]")
    axs[0, 0].set_ylabel("Amplitude")
    axs[0, 0].grid()
    
    if plot_zoom:
        axs[0, 1].set_title(s_name)
        axs[0, 1].plot(t, s, color='C0')
        axs[0, 1].set_xlabel("Time [s]")
        axs[0, 1].set_ylabel("Amplitude")
        axs[0, 1].set_xlim(xlim)
        axs[0, 1].grid()
    else:
        axs[0, 1].remove()

    # plot different spectrum types:
    axs[1, 0].set_title("Magnitude Spectrum")
    axs[1, 0].magnitude_spectrum(s, Fs=Fs, color='C1')
    axs[1, 0].grid()

    axs[1, 1].set_title("Log. Magnitude Spectrum")
    axs[1, 1].magnitude_spectrum(s, Fs=Fs, scale='dB', color='C1')
    axs[1, 1].grid()

    axs[2, 0].set_title("Phase Spectrum ")
    axs[2, 0].phase_spectrum(s, Fs=Fs, color='C2')
    axs[2, 0].grid()

    axs[2, 1].set_title("Angle Spectrum")
    axs[2, 1].angle_spectrum(s, Fs=Fs, color='C2')
    axs[2, 1].grid()

    fig.tight_layout()
    plt.show()

def plot_recorded_signal(sig, sample_rate=41000):
    max_time = (sig.size/sample_rate) 
    time = np.linspace(0, max_time, sig.size) 
    plt.plot(time, sig)
    plt.ylabel('amplitude')
    plt.xlabel('time [ms]')
    plt.grid()
    plt.show()


def saveBufferToCSV(buffer, pathToFile):
    df = pd.DataFrame(buffer)
    df.to_csv(pathToFile, index=False)


def readCSVData(pathToFile):
    df = pd.read_csv(pathToFile)
    df.dropna(inplace=True)
    return df.to_numpy()[:, 1]

def readCSVData_(pathToFile):
    df = pd.read_csv(pathToFile)
    df.dropna(inplace=True)
    return df.to_numpy()[:, 0]

# def test_run():
#     num_captured = 5
#     impulse_responses = []
    
#     mls = readCSVData('/Users/hugo/Desktop/dev/easyeyes/python-flask-server/data/MLS.csv');
#     #plot_frequency_spectrum(mls, 'MLS', True)
    
#     sig = None
#     for i in range(num_captured):
#         sig = readCSVData(f'/Users/hugo/Desktop/SC-Captures/07-17-22-96000-5/recordedMLSignal_{i}.csv')
#         #plot_recorded_signal(sig)
        
#         ir = run_ir_task(sig, debug=True)
#         #plot_frequency_spectrum(ir, 'Impulse Reponse')
#         impulse_responses.append(ir)
#         #plot_recorded_signal(impulse_responses[i])
        
#     plot_frequency_spectrum(sig, 'Recorded MLS', True, [2.005, 2.01])

#     smallest = np.Inf
#     for ir in impulse_responses:
#         if len(ir) < smallest:
#             smallest = len(ir)
    
#     for i, ir in enumerate(impulse_responses):
#         impulse_responses[i] = ir[0:smallest]
    
#     ir = np.mean(impulse_responses, axis=0)
#     plot_frequency_spectrum(ir, 'Impulse Response', True, [-0.025, 0.05])
#     saveBufferToCSV(ir.real, '/Users/hugo/Desktop/ir_py.csv')
    
#     g = run_iir_task(impulse_responses, debug=True)
#     g = np.array(g)
#     plot_frequency_spectrum(g, 'Inverted Impulse Response', True, [-0.000025, 0.0005])
#     # plot_recorded_signal(g)
    
#     saveBufferToCSV(2.*(g - np.min(g))/np.ptp(g)-1, '/Users/hugo/Desktop/iir_py.csv')
    
#     # convolved = convolve(g, sig)
#     # plot_recorded_signal(convolved)
    
#     # corrected = convolve(convolved, impulse_responses[len(impulse_responses)-1])
    
#     # plot_recorded_signal(corrected)

    
#     # for sig in recordedSignals:
#     #     plot_recorded_signal(convolve(sig, g))
    
#     recovered = recover_signal(s=mls, g=g, h=ir)
#     plot_frequency_spectrum(recovered, 'Recovered MLS')

def convolve_wav():
    sampleRate, rawData = wavfile.read("/Users/hugo/Desktop/dev/easyeyes/speaker-calibration/dist/example/Queen-Bohemian_Rhapsody.wav")
    g = readCSVData_('/Users/hugo/Desktop/iir_py.csv')
    h = readCSVData_('/Users/hugo/Desktop/ir_py.csv')
    
    waveFile_left = np.array(rawData[:, 0], dtype=float)
    waveFile_right = np.array(rawData[:, 1], dtype=float)
    # waveFile = waveFile[:131002]
        
    #plot_frequency_spectrum(waveFile, 'WAV File')
    
    #recovered = recover_signal(s=waveFile, g=g, h=h)
    
    waveFile_left_g_filtered = g_filter(waveFile_left, g)
    waveFile_right_g_filtered = g_filter(waveFile_right, g)
    
    new_waveFile = np.transpose(np.array([waveFile_left_g_filtered.real, waveFile_right_g_filtered.real]))
    wavfile.write('/Users/hugo/Desktop/dev/easyeyes/speaker-calibration/dist/example/Queen-Bohemian_Rhapsody_g_filtered.wav', sampleRate, new_waveFile.astype(np.int16))
    
    #plot_frequency_spectrum(waveFile_left_g_filtered, 'g Filtered WAV File')
    #plot_frequency_spectrum(recovered, 'Recovered WAV File')

def allHzPowerCheck(rec, fs, _calibrateSoundPowerBinDesiredSec, _calibrateSoundBurstSec):
    coarseHz = 1 / _calibrateSoundPowerBinDesiredSec 
    power = np.square(np.array(rec))
    # Adjust coarseHz so that fs is an integer
    # multiple of coarseHz.
    n = int(round(fs / coarseHz))
    coarseHz = int(fs / n)
    # Sampling times for plotting
    t = np.arange(len(power)) / fs
    coarseSamples = int(np.ceil(len(power) / n))
    coarsePowerDb = np.zeros(coarseSamples)
    coarseT = np.zeros(coarseSamples)
    for i in range(coarseSamples):
      indices = range(i * n, min((i + 1) * n, len(power)))
      extremeIndices = [indices[0],indices[-1]]
      coarsePowerDb[i] = 10 * np.log10(np.mean(power[indices]))
      coarseT[i] = np.mean(t[extremeIndices])
    prepSamples=round(coarseHz * _calibrateSoundBurstSec)
    sdDb=np.round(np.std(coarsePowerDb[prepSamples:]),1)
    coarseT = np.round(coarseT, 1).tolist()
    coarsePowerDb = np.round(coarsePowerDb,1).tolist()
    start = np.interp(_calibrateSoundBurstSec,coarseT,coarsePowerDb)
    warmupT = coarseT[:prepSamples]
    warmupDb = coarsePowerDb[:prepSamples]
    if warmupT[-1] < _calibrateSoundBurstSec:
        warmupT = warmupT + [_calibrateSoundBurstSec]
        warmupDb = warmupDb + [start]
    recT = coarseT[prepSamples:]
    recDb = coarsePowerDb[prepSamples:]
    if recT[0] > _calibrateSoundBurstSec:
        recT = [_calibrateSoundBurstSec] + coarseT[prepSamples:]
        recDb = [start] + coarsePowerDb[prepSamples:]
    return warmupT, warmupDb, recT, recDb, sdDb

def volumePowerCheck(rec, fs, preSec, Sec, _calibrateSoundPowerBinDesiredSec):
    coarseHz = 1 / _calibrateSoundPowerBinDesiredSec 
    power = np.square(np.array(rec))
    # Adjust coarseHz so that fs is an integer
    # multiple of coarseHz.
    n = int(round(fs / coarseHz))
    coarseHz = int(fs / n)
    # Sampling times for plotting
    t = np.arange(len(power)) / fs
    coarseSamples = int(np.ceil(len(power) / n))
    coarsePowerDb = np.zeros(coarseSamples)
    coarseT = np.zeros(coarseSamples)
    for i in range(coarseSamples):
      indices = range(i * n, min((i + 1) * n, len(power)))
      extremeIndices = [indices[0],indices[-1]]
      coarsePowerDb[i] = 10 * np.log10(np.mean(power[indices]))
      coarseT[i] = np.mean(t[extremeIndices])
    
    prepSamples=round(coarseHz * preSec)
    postSamples=round(coarseHz * (preSec + Sec))
    sdDb=np.round(np.std(coarsePowerDb[prepSamples:]),1)
    coarseT = np.round(coarseT, 1).tolist()
    coarsePowerDb = np.round(coarsePowerDb,1).tolist()
    start = np.interp(preSec,coarseT,coarsePowerDb)
    end = np.interp((preSec + Sec),coarseT,coarsePowerDb)
    preT = coarseT[:prepSamples]
    preDb = coarsePowerDb[:prepSamples]
    if preT[-1] < preSec:
        preT = coarseT[:prepSamples] + [preSec]
        preDb = coarsePowerDb[:prepSamples] + [start]
    recT = coarseT[prepSamples:postSamples]
    recDb = coarsePowerDb[prepSamples:postSamples]
    if recT[0] > preSec:
        recT = [preSec] + coarseT[prepSamples:postSamples]
        recDb = [start] + coarsePowerDb[prepSamples:postSamples]
    if rec[-1] < (preSec + Sec):
        recT = coarseT[prepSamples:postSamples] + [(preSec + Sec)]
        recDb = coarsePowerDb[prepSamples:postSamples] + [end]
    postT = coarseT[postSamples:]
    postDb = coarsePowerDb[postSamples:]
    if postT[0] > (preSec + Sec):
        postT = [(preSec + Sec)] + coarseT[postSamples:]
        postDb = [end] + coarsePowerDb[postSamples:]
    return preT, preDb, recT, recDb, postT, postDb, sdDb

if __name__ == '__main__':
    #test_run()
    convolve_wav()