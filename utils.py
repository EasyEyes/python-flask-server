import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from impulse_response import run_ir_task
from scipy.signal import convolve
from pympler.tracker import SummaryTracker
import gc



parser = argparse.ArgumentParser(description="Ingest audio samples from client.")
parser.add_argument("--sampleRate", type=int, help="Sample rate of payload, in hz")
parser.add_argument("--payload", nargs="+", type=float, help="Array of audio samples")
parser.add_argument("--debug", action="store_true", help="Print debug info")

def plot_power_spectrum(sig, sample_rate=96000):
    sampling_rate = 30.0

    time = np.arange(0, len(sig), 1/sampling_rate)

    fourier_transform = np.fft.rfft(sig)

    abs_fourier_transform = np.abs(fourier_transform)

    power_spectrum = np.square(abs_fourier_transform)

    frequency = np.linspace(0, sampling_rate/2, len(power_spectrum))

    plt.plot(frequency, power_spectrum)
    plt.show()


def saveBufferToCSV(buffer, pathToFile):
    df = pd.DataFrame(buffer)
    df.to_csv(pathToFile)


def readCSVData(pathToFile):
    df = pd.read_csv(pathToFile)
    df.dropna(inplace=True)
    return df.to_numpy()[:, 1]

def test_run():
    recordedSignal = [readCSVData('/Users/hugo/Desktop/dev/easyeyes/python-flask-server/data/recordedMLSignal_4.csv')]
    tracker = SummaryTracker()    
    g = run_ir_task(recordedSignal)
        
    time = np.arange(0, 10, 1/96000)
    data = np.sin(2*np.pi*6*time) + np.random.randn(len(time))
    # print(len(time), len(data))
    sig = convolve(data, g)
    gc.collect(2)
    plot_power_spectrum(sig)
    tracker.print_diff()

if __name__ == '__main__':
    test_run()