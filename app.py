import os
import tracemalloc

import psutil

import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use("Agg")
import time
from flask import Flask, request, make_response
from flask_cors import CORS, cross_origin
from impulse_response import run_ir_task
from inverted_impulse_response import run_component_iir_task, run_system_iir_task
from volume import run_volume_task,run_volume_task_nonlinear
from volume import get_model_parameters
import numpy as np
from scipy.fft import fft, ifft, irfft, fftfreq
from scipy.interpolate import interp1d
from scipy.signal import max_len_seq
import math

app = Flask(__name__)
CORS(app, resources = {r"/*": {"origins": "*"}})

process = psutil.Process(os.getpid())
tracemalloc.start()

def handle_impulse_response_task(request_json, task):
    start_time = time.time()
    if "payload" not in request_json:
        return 400, "Request Body is missing a 'payload' entry"
    if "sample-rate" not in request_json:
        return 400, "Request Body us missing a 'sample-rate' entry"
    if "P" not in request_json:
        return 400, "Request Body us missing a 'P' entry"
    if "mls" not in request_json:
        return 400, "Request Body is missing a 'mls' entry"
    if "numPeriods" not in request_json:
        return 400, "Request Body is missing a 'numPeriods' entry"
    
    recordedSignalsJson = request_json["payload"]
    mls = request_json["mls"]
    sampleRate = request_json["sample-rate"]
    P = request_json["P"]
    NUM_PERIODS = request_json["numPeriods"]
    NUM_PERIODS = int(NUM_PERIODS)
    print("Starting IR Task")
    ir, autocorrelation = run_ir_task(mls,recordedSignalsJson, P, sampleRate,NUM_PERIODS)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"============== handle_impulse_response task, time taken: {elapsed_time}s ==============")
    return 200, {
        str(task): {
            'ir':ir,
            'autocorrelation': autocorrelation
        }
    }

def handle_component_inverse_impulse_response_task(request_json, task):
    if "payload" not in request_json:
        return 400, "Request Body is missing a 'payload' entry"
    if "mls" not in request_json:
        return 400, "Request Body is missing a 'mls' entry"
    if "lowHz" not in request_json:
        return 400, "Request Body is missing a 'lowHz' entry"
    if "highHz" not in request_json:
        return 400, "Request Body is missing a 'highHz' entry"
    if "componentIRGains" not in request_json:
        return 400, "Request body is missing a 'componentIRGains'"
    if "componentIRFreqs" not in request_json:
        return 400, "Request body is missing a 'componentIRFreqs'"
    if "sampleRate" not in request_json:
        return 400, "Request body is missing a 'sampleRate'"
    if "iirLength" not in request_json:
        return 400, "Request body is missing a 'iirLength'"
    if "num_periods" not in request_json:
        return 400, "Request body is missing a 'num_periods'"
    start_time = time.time()
    impulseResponsesJson = request_json["payload"]
    iir_length = request_json["iirLength"]
    mls = request_json["mls"]
    lowHz = request_json["lowHz"]
    highHz = request_json["highHz"]
    componentIRGains = request_json["componentIRGains"]
    componentIRFreqs = request_json["componentIRFreqs"]
    sampleRate = request_json["sampleRate"]
    num_periods = request_json["num_periods"]
    calibrateSoundBurstDb = request_json["calibrateSoundBurstDb"]
    result, convolution, ir,frequencies, iir_no_bandpass, ir_time = run_component_iir_task(impulseResponsesJson,mls,lowHz,highHz,iir_length,componentIRGains,componentIRFreqs,num_periods,sampleRate, calibrateSoundBurstDb)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"============== component_inverse_impulse_response task, time taken: {elapsed_time}s ==============")
    return 200, {
        str(task): {
                        "iir":result,
                        "convolution":convolution,
                        "ir":ir,
                        "frequencies":frequencies,
                        "iirNoBandpass":iir_no_bandpass,
                        "irTime": ir_time,
                    }
    }

def handle_system_inverse_impulse_response_task(request_json, task):
    start_time = time.time()
    if "payload" not in request_json:
        return 400, "Request Body is missing a 'payload' entry"
    if "mls" not in request_json:
        return 400, "Request Body is missing a 'mls' entry"
    if "lowHz" not in request_json:
        return 400, "Request Body is missing a 'lowHz' entry"
    if "highHz" not in request_json:
        return 400, "Request Body is missing a 'highHz' entry"
    if "sampleRate" not in request_json:
        return 400, "Request body is missing a 'sampleRate'"
    if "iirLength" not in request_json:
        return 400, "Request body is missing a 'iirLength'"
    if "num_periods" not in request_json:
        return 400, "Request body is missing a 'num_periods'"

    impulseResponsesJson = request_json["payload"]
    iir_length = request_json["iirLength"]
    mls = request_json["mls"]
    lowHz = request_json["lowHz"]
    highHz = request_json["highHz"]
    sampleRate = request_json["sampleRate"]
    num_periods = request_json["num_periods"]
    calibrateSoundBurstDb = request_json["calibrateSoundBurstDb"]
    result, convolution, ir, iir_no_bandpass = run_system_iir_task(impulseResponsesJson,mls,lowHz,iir_length,highHz,num_periods,sampleRate, calibrateSoundBurstDb)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"============== system_inverse_impulse_response task, time taken: {elapsed_time}s ==============")
    return 200, {
        str(task): {
                        "iir":result,
                        "convolution":convolution,
                        "ir":ir,
                        "iirNoBandpass":iir_no_bandpass
                    }
    }

def handle_volume_task(request_json, task):
    start_time = time.time()
    if "payload" not in request_json:
        return 400, "Request Body is missing a 'payload' entry"
    if "sample-rate" not in request_json:
        return 400, "Request Body us missing a 'sample-rate' entry"
    recordedSignalJson = request_json["payload"]
    sampleRate = request_json["sample-rate"]
    soundGainDbSPL, _, _, _ = run_volume_task(recordedSignalJson, sampleRate)
    return 200, {
        str(task): soundGainDbSPL
    }

def handle_volume_task_nonlinear(request_json, task):
    start_time = time.time()
    if "payload" not in request_json:
        return 400, "Request Body is missing a 'payload' entry"
    if "sample-rate" not in request_json:
        return 400, "Request Body us missing a 'sample-rate' entry"
    recordedSignalJson = request_json["payload"]
    sampleRate = request_json["sample-rate"]
    lCalib = request_json["lCalib"]
    soundGainDbSPL, P, L, _, L1000, P1000, thd, rms, soundGainDbSPL1000 = run_volume_task_nonlinear(recordedSignalJson, sampleRate) #L is outDbSPL
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"==============  volume_task_nonlinear task, time taken: {elapsed_time}s ==============")
    return 200, {
        str(task): {
            "outDbSPL":L,
            "outDbSPL1000": float(L1000),
            "thd": thd,
        }
    }

def handle_volume_parameters(request_json,task):
    start_time = time.time()
    inDB = request_json["inDBValues"]
    outDBSPL = request_json["outDBSPLValues"]
    lCalib = request_json["lCalib"]
    componentGainDBSPL = request_json["componentGainDBSPL"]
    backgroundDBSPL, gainDBSPL, T, R, W, rmsError, modelGuesses = get_model_parameters(inDB,outDBSPL,lCalib,componentGainDBSPL)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"============== handle_volume_parameters task, time taken: {elapsed_time}s ==============")
    return 200, {
        str(task): {
            "backgroundDBSPL":backgroundDBSPL,
            "gainDBSPL":gainDBSPL,
            "T":T,
            "R":R,
            "W":W,
            "RMSError":rmsError,
            "initialGuesses":{
                "backgroundDBSPL": modelGuesses[0],
                "gainDBSPL": modelGuesses[1],
                "T": modelGuesses[2],
                "R": modelGuesses[3],
                "W": modelGuesses[4],
            }
            }
    }

def handle_psd_task(request_json,task):
    start_time = time.time()
    rec_unconv = request_json["unconv_rec"]
    rec_conv = request_json["conv_rec"]
    sampleRate = request_json["sampleRate"]
    print('length of rec')
    print(len(rec_unconv))
    [y_unconv, x_unconv] = plt.psd(rec_unconv,Fs=sampleRate,NFFT=2048,scale_by_freq=False)
    [y_conv,x_conv] = plt.psd(rec_conv, Fs=sampleRate, NFFT=2048,scale_by_freq=False)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"================ psd_task, time taken: {elapsed_time}s ================")
    return 200, {
        str(task): {
            "x_unconv":x_unconv.tolist(),
            "y_unconv":y_unconv.tolist(),
            "x_conv":x_conv.tolist(),
            "y_conv":y_conv.tolist(),
            }
    }
def handle_background_psd_task(request_json,task):
    start_time = time.time()
    background_rec = request_json["background_rec"]
    sampleRate = request_json["sampleRate"]

    [y_background, x_background] = plt.psd(background_rec,Fs=sampleRate,NFFT=2048,scale_by_freq=False)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"================ handle_background_psd task, time taken: {elapsed_time}s ================")
    return 200, {
        str(task): {
            "x_background":x_background.tolist(),
            "y_background":y_background.tolist(),
            }
    }

def handle_mls_psd_task(request_json,task):
    start_time = time.time()
    mls = request_json["mls"]
    sampleRate = request_json["sampleRate"]
    [y_mls, x_mls] = plt.psd(mls,Fs=sampleRate,NFFT=2048,scale_by_freq=False)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"================ handle_mls_psd task, time taken: {elapsed_time}s ================")
    return 200, {
        str(task): {
            "x_mls":x_mls.tolist(),
            "y_mls":y_mls.tolist(),
            }
    }

def handle_mls_task(request_json,task):
    start_time = time.time()
    #length of mls will be 2**nbits - 1
    desired_length = request_json["length"]
    calibrateSoundBurstDb = request_json["calibrateSoundBurstDb"]
    nbits = math.ceil(math.log(desired_length + 1, 2))
    ret_arr = max_len_seq(nbits,length=desired_length)
    mls = ret_arr[0]
    mls_transformed = np.where(mls == 0, -1, 1)
    scaled_mls_transformed = mls_transformed * calibrateSoundBurstDb
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"================ handle_mls task, time taken: {elapsed_time}s ================")
    return 200, {
        str(task):{
            "mls": scaled_mls_transformed.tolist(),
            "unscaledMLS": mls_transformed.tolist()
        }
    }

def handle_subtracted_psd_task(request_json,task):
    start_time = time.time()
    #print(request_json);
    rec = request_json["rec"]
    # knownGain = request_json["knownGains"]
    # knownFreq = request_json["knownFrequencies"]
    sample_rate = request_json["sampleRate"]
    # rec_fft = fft(rec)
    # num_samples = len(rec)
    # frequencies = fftfreq(num_samples,1/sample_rate)
    #interpolation part
    #1) convert rec_fft to dB
    # rec_fft_db = 20*np.log10(abs(rec_fft))
    #2) interpolate and subtract
    #interpolate function for componentGains and componentFreqs
    # interp_func = interp1d(knownFreq,knownGain)
    #3) some sorting to make sure gains and frequencies are in sorted order when returned
    # min_freq = min(knownFreq)
    # max_freq = max(knownFreq)
    # inbounds_indices = np.where((abs(frequencies) >= min_freq) & (abs(frequencies) <= max_freq))
    # outbounds_indices = np.where((abs(frequencies) < min_freq) | (abs(frequencies) > max_freq))
    # inbounds_frequencies = abs(frequencies[inbounds_indices])
    # inbounds_rec_fft_db = rec_fft_db[inbounds_indices]
    # interp_gain2 = interp_func(inbounds_frequencies)
    # result = inbounds_rec_fft_db - interp_gain2
    # final_result = np.zeros_like(frequencies)
    # final_result[inbounds_indices] = result
    # final_result[outbounds_indices] = rec_fft_db[outbounds_indices]
    #3) convert rec_fft to linear, invert back to time
    # rec = 10**(final_result/20)
    # rec = ifft(rec)
    [y, x] = plt.psd(rec,Fs=sample_rate,NFFT=2048,scale_by_freq=False)
    #[x_conv,y_conv] = plt.psd(rec_conv, Fs=96000, scale_by_freq=False)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"================ handle psd task, time taken: {elapsed_time}s ================")
    return 200, {
        str(task): {
            "x":x.tolist(),
            "y":y.tolist(),
            }
    }


SUPPORTED_TASKS = {
    'impulse-response': handle_impulse_response_task,
    'component-inverse-impulse-response': handle_component_inverse_impulse_response_task,
    'system-inverse-impulse-response': handle_system_inverse_impulse_response_task,
    'volume': handle_volume_task_nonlinear,
    'volume-parameters': handle_volume_parameters,
    'psd': handle_psd_task,
    'subtracted-psd':handle_subtracted_psd_task,
    'mls':handle_mls_task,
    'background-psd': handle_background_psd_task,
    'mls-psd': handle_mls_psd_task
}

@app.route("/task/<string:task>", methods=['POST'])
@cross_origin()
def task_handler(task):
    if task not in SUPPORTED_TASKS:
        return 'ERROR'
    content_type = request.headers.get('Content-Type')
    if (content_type == 'application/json'):
        json = request.get_json()
        headers = {"Content-Type": "application/json"}
        status, result = SUPPORTED_TASKS[task](json, task)
        resp = make_response(result, status)
        resp.headers = headers
        return resp
    else:
        return 'Content-Type not supported'
    
@app.route('/memory')
@cross_origin()
def print_memory():
    return {'memory': process.memory_info().rss}    

@app.route("/snapshot")
@cross_origin()
def snap():
    global s
    if not s:
        s = tracemalloc.take_snapshot()
        return "taken snapshot\n"
    else:
        lines = []
        top_stats = tracemalloc.take_snapshot().compare_to(s, 'lineno')
        for stat in top_stats[:5]:
            lines.append(str(stat))
        return "\n".join(lines)

if __name__ == '__main__':
    app.run()
