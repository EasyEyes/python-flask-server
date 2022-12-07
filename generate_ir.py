import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from impulse_response import run_ir_task, compute_impulse_resp

parser = argparse.ArgumentParser(description="Ingest audio samples from client.")
parser.add_argument("--sampleRate", type=int, help="Sample rate of payload, in hz")
parser.add_argument("--payload", nargs="+", type=float, help="Array of audio samples")
parser.add_argument("--debug", action="store_true", help="Print debug info")


def plot_frequency_spectrum(s, s_name, xlim=[0, 0.01], dt=96000):
    max_time = (s.size/dt) 
    t = np.linspace(0, max_time, s.size) 
    Fs = 1 / dt  # sampling frequency

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 7))
    print(fig, axs, s, t)


    axs[0].set_title(s_name)
    axs[0].plot(t, s, color='C0')
    axs[0].set_xlabel("Time [s]")
    axs[0].set_ylabel("Amplitude")
    axs[0].set_xlim(xlim)
    axs[0].grid()

    axs[1].set_title("Log. Magnitude Spectrum")
    axs[1].magnitude_spectrum(s, Fs=Fs, scale='dB', color='C1')
    axs[1].grid()

    fig.tight_layout()
    plt.show()



def readCSVData(pathToFile):
    df = pd.read_csv(pathToFile)
    df.dropna(inplace=True)
    return df.to_numpy()[:, 1]

def readCSVData_(pathToFile):
    df = pd.read_csv(pathToFile)
    df.dropna(inplace=True)
    return df.to_numpy()[:, 0]

def saveBufferToCSV(buffer, pathToFile):
    df = pd.DataFrame(buffer)
    df.to_csv(pathToFile, index=False)

def test_run():
    num_captured = 1
    impulse_responses = []

    sig = None
    for i in range(num_captured):
        sig = readCSVData(f'/home/billy/Documents/EasyEyes/presentation/data/MLS.csv')
        #sig = readCSVData(f'/home/billy/Documents/EasyEyes/presentation/data/recordedMLSignal_0.csv')
        #print(len(sig))
        #sig = sig.append(sig).append(sig)
        #ir = run_ir_task(sig, debug=True)
        ir = compute_impulse_resp(np.array(sig,dtype=np.float32),(1 << 18) -1, 96000) 
        #print(ir)
        saveBufferToCSV(ir.real,f'/home/billy/Documents/EasyEyes/presentation/data/ir_from_MLS_compute.csv')


    #smallest = np.Inf
    #for ir in impulse_responses:
    #    if len(ir) < smallest:
    #        smallest = len(ir)
    #print(smallest, 'hello')

    #for i, ir in enumerate(impulse_responses):
    #    impulse_responses[i] = ir[0:smallest]

    #ir = np.mean(impulse_responses, axis=0)
    #plot_frequency_spectrum(ir, 'Impulse Response')
    #saveBufferToCSV(ir.real, '/home/billy/Documents/EasyEyes/ivan/data/ir_from_MLS_conv_ones.csv')

if __name__ == '__main__':
    test_run()