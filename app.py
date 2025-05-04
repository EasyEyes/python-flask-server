import os
import tracemalloc
import gc
import psutil
import json

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import time
from flask import Flask, request, make_response
from flask_cors import CORS, cross_origin
from impulse_response import run_ir_task, estimate_samples_per_mls_, adjust_mls_length, compute_impulse_resp, impulse_to_frequency_response
from inverted_impulse_response import run_component_iir_task, run_system_iir_task, run_convolution_task, run_ir_convolution_task
from volume import run_volume_task,run_volume_task_nonlinear
from volume import get_model_parameters
import numpy as np
from scipy.signal import max_len_seq
from scipy.fft import fft
from utils import allHzPowerCheck, volumePowerCheck
import math

app = Flask(__name__)
CORS(app, resources = {r"/*": {"origins": "*"}})

process = psutil.Process(os.getpid())
tracemalloc.start()


def handle_autocorrelation_task(request_json, task):
    if "payload" not in request_json:
        return 400, "Request Body is missing a 'payload' entry"
    if "sample-rate" not in request_json:
        return 400, "Request Body us missing a 'sample-rate' entry"
    if "mls" not in request_json:
        return 400, "Request Body is missing a 'mls' entry"
    if "numPeriods" not in request_json:
        return 400, "Request Body is missing a 'numPeriods' entry"
    sig = request_json["payload"]
    mls = request_json["mls"]
    sampleRate = request_json["sample-rate"]
    NUM_PERIODS = int(request_json["numPeriods"])
    print("number of period ", NUM_PERIODS)
    sig = np.array(sig, dtype=np.float32)
    MLS = fft(np.array(mls, dtype=np.float32))
    L = len(MLS)
    fs2, L_new_n, dL_n, autocorrelation = estimate_samples_per_mls_(sig, NUM_PERIODS, sampleRate, L)
    print_memory_usage()
    gc.collect()
    return 200, {
        str(task): {
            'autocorrelation': autocorrelation.tolist(),
            'fs2': fs2.tolist(),
            "L_new_n": L_new_n.tolist(), 
            "dL_n": dL_n.tolist()
        }
    }

def handle_impulse_response_task(request_json, task):
    start_time = time.time()
    if "sample-rate" not in request_json:
        return 400, "Request Body us missing a 'sample-rate' entry"
    if "mls" not in request_json:
        return 400, "Request Body is missing a 'mls' entry"
    if "numPeriods" not in request_json:
        return 400, "Request Body is missing a 'numPeriods' entry"
    if "L_new_n" not in request_json:
        return 400, "Request Body is missing a 'L_new_n' entry"
    if "dL_n" not in request_json:
        return 400, "Request Body is missing a 'dL_n' entry"
    if "sig" not in request_json:
        return 400, "Request Body is missing a 'sig' entry"
    if "fs2" not in request_json:
        return 400, "Request Body is missing a 'fs2' entry"
    MLS = fft(np.array(request_json['mls'], dtype=np.float32))
    L = len(MLS)
    sig = np.array(request_json["sig"], dtype=np.float32)
    L_new_n = request_json["L_new_n"]
    dL_n = request_json["dL_n"]
    NUM_PERIODS = request_json["numPeriods"]
    fs2 = request_json["fs2"]
    NUM_PERIODS = int(NUM_PERIODS)
    print("Starting IR Task")
    OUT_MLS2_n = adjust_mls_length(sig, NUM_PERIODS, L, L_new_n, dL_n)
    ir = compute_impulse_resp(MLS, OUT_MLS2_n, L, fs2, NUM_PERIODS)
    print_memory_usage()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"============== handle_impulse_response task, time taken: {elapsed_time}s ==============")
    return 200, {
        str(task): {
            'ir':ir.tolist()
        }
    }


def handle_all_hz_power_check_task(request_json, task):
    recordedSignalsJson = request_json["payload"]
    sampleRate = request_json["sampleRate"]
    binDesiredSec = request_json["binDesiredSec"]
    burstSec = request_json["burstSec"]
    repeats = request_json["repeats"]
    warmUp = request_json["warmUp"] # number of warm up PERIOD
    warmupT, warmupDb, recT, recDb, sd, postT, postDb  = allHzPowerCheck(recordedSignalsJson, sampleRate, binDesiredSec, burstSec,repeats, warmUp)
    print_memory_usage()
    gc.collect()
    return 200, {
        str(task): {
            'sd':sd,
            'warmupT': warmupT, 
            'warmupDb': warmupDb, 
            'recT': recT, 
            'recDb': recDb,
            'postT': postT,
            'postDb': postDb
        }
    }

def handle_volume_power_check_task(request_json, task):
    recordedSignalsJson = request_json["payload"]
    sampleRate = request_json["sampleRate"]
    preSec = request_json["preSec"]
    Sec = request_json["Sec"]
    binDesiredSec = request_json["binDesiredSec"]
    preT, preDb, recT, recDb, postT, postDb, sd = volumePowerCheck(recordedSignalsJson, sampleRate, preSec, Sec, binDesiredSec)
    return 200, {
        str(task): {
            'sd':sd,
            'preT': preT, 
            'preDb': preDb, 
            'recT': recT, 
            'recDb': recDb,
            'postT': postT,
            'postDb': postDb
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
    if "mlsAmplitude" not in request_json:
        return 400, "Request body is missing a 'mlsAmplitude'"
    if "irLength" not in request_json:
        return 400, "Request body is missing a 'irLength'"
    if "calibrateSoundSmoothOctaves" not in request_json:
        return 400, "Request body is missing a 'calibrateSoundSmoothOctaves'"
    if "calibrateSoundSmoothMinBandwidthHz" not in request_json:
        return 400, "Request body is missing a 'calibrateSoundSmoothMinBandwidthHz'"
    if "calibrateSoundBurstFilteredExtraDb" not in request_json:
        return 400, "Request body is missing a 'calibrateSoundBurstFilteredExtraDb'"
    start_time = time.time()
    impulseResponsesJson = request_json["payload"]
    iir_length = request_json["iirLength"]
    mls = request_json["mls"]
    lowHz = request_json["lowHz"]
    highHz = request_json["highHz"]
    componentIRGains = request_json["componentIRGains"]
    componentIRFreqs = request_json["componentIRFreqs"]
    sampleRate = request_json["sampleRate"]
    mls_amplitude = request_json["mlsAmplitude"]
    irLength = request_json["irLength"]
    calibrateSoundSmoothOctaves = request_json["calibrateSoundSmoothOctaves"]
    calibrateSoundSmoothMinBandwidthHz = request_json["calibrateSoundSmoothMinBandwidthHz"]
    calibrate_sound_burst_filtered_extra_db = request_json["calibrateSoundBurstFilteredExtraDb"]
    _calibrateSoundIIRPhase = request_json["calibrateSoundIIRPhase"]
    result, ir,frequencies, iir_no_bandpass, ir_time, angle, ir_origin, system_angle, attenuatorGain_dB, fMaxHz = run_component_iir_task(impulseResponsesJson,mls,lowHz,highHz,iir_length,componentIRGains,componentIRFreqs,sampleRate, mls_amplitude, irLength, calibrateSoundSmoothOctaves,calibrateSoundSmoothMinBandwidthHz, calibrate_sound_burst_filtered_extra_db, _calibrateSoundIIRPhase)
    print_memory_usage()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"============== component_inverse_impulse_response task, time taken: {elapsed_time}s ==============")
    return 200, {
        str(task): {
                        "iir":result,
                        "ir":ir,
                        "frequencies":frequencies,
                        "iirNoBandpass":iir_no_bandpass,
                        "irTime": ir_time,
                        "component_angle": angle,
                        "system_angle":system_angle,
                        "irOrigin": ir_origin,
                        "attenuatorGain_dB":attenuatorGain_dB,
                        "fMaxHz":fMaxHz
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
    if "calibrateSoundBurstFilteredExtraDb" not in request_json:
        return 400, "Request body is missing a 'calibrateSoundBurstFilteredExtraDb'"
    impulseResponsesJson = request_json["payload"]
    iir_length = request_json["iirLength"]
    mls = request_json["mls"]
    lowHz = request_json["lowHz"]
    highHz = request_json["highHz"]
    sampleRate = request_json["sampleRate"]
    mls_amplitude = request_json["mlsAmplitude"]
    calibrate_sound_burst_filtered_extra_db = request_json["calibrateSoundBurstFilteredExtraDb"]
    _calibrateSoundIIRPhase = request_json["calibrateSoundIIRPhase"]
    result, ir, iir_no_bandpass, attenuatorGain_dB, fMaxHz = run_system_iir_task(impulseResponsesJson,mls,lowHz,iir_length,highHz,sampleRate, mls_amplitude,calibrate_sound_burst_filtered_extra_db, _calibrateSoundIIRPhase)
    print_memory_usage()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"============== system_inverse_impulse_response task, time taken: {elapsed_time}s ==============")
    return 200, {
        str(task): {
                        "iir":result,
                        "ir":ir,
                        "iirNoBandpass":iir_no_bandpass,
                        "attenuatorGain_dB":attenuatorGain_dB,
                        "fMaxHz":fMaxHz
                    }
    }

def handle_convolution_task(request_json, task):
     if "mls" not in request_json:
         return 400, "Request Body is missing a 'mls' entry"
     if "inverse_response" not in request_json:
         return 400, "Request body is missing a 'inverse_response'"
     if "inverse_response_no_bandpass" not in request_json:
         return 400, "Request body is missing a 'inverse_response_no_bandpass'"
     if "attenuatorGain_dB" not in request_json:
         return 400, "Request body is missing a 'attenuatorGain_dB'"
     if "mls_amplitude" not in request_json:
         return 400, "Request body is missing a 'mls_amplitude'"
     mls = request_json['mls']
     inverse_response = request_json['inverse_response']
     inverse_response_no_bandpass = request_json['inverse_response_no_bandpass']
     attenuatorGain_dB = request_json['attenuatorGain_dB']
     mls_amplitude = request_json['mls_amplitude']
     conv, conv_nbp = run_convolution_task(inverse_response, mls, inverse_response_no_bandpass, attenuatorGain_dB, mls_amplitude)
     gc.collect()
     return 200, {
         str(task): {
             'convolution': conv,
             'convolution_no_bandpass': conv_nbp
             }
             }

def handle_volume_task(request_json, task):
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
    print("soundGainDbSPL", soundGainDbSPL)
    print("outDbSPL", L)
    print("outDbSPL1000", L1000)
    print("thd", thd)
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
    [y_unconv, x_unconv] = plt.psd(rec_unconv,Fs=sampleRate, window=plt.mlab.window_none, NFFT=2048,scale_by_freq=False)
    [y_conv,x_conv] = plt.psd(rec_conv, Fs=sampleRate, window=plt.mlab.window_none, NFFT=2048,scale_by_freq=False)
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
    [y_mls, x_mls] = plt.psd(mls,Fs=sampleRate, window=plt.mlab.window_none, NFFT=2048,scale_by_freq=False)
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
    calibrateSoundBurstMLSVersions = int(request_json['calibrateSoundBurstMLSVersions'])
    desired_length = request_json["length"]
    amplitude = request_json["amplitude"]
    mls_transformed = []
    scaled_mls_transformed = []
    nbits = math.ceil(math.log(desired_length + 1, 2))
    ret_arr = max_len_seq(nbits,length=desired_length*calibrateSoundBurstMLSVersions)
    mls = ret_arr[0]
    for i in range(calibrateSoundBurstMLSVersions):
         print(i)
         mls_transformed.append((np.where(mls == 0, -1, 1)[i*desired_length:(i+1)*desired_length]).tolist())
         scaled_mls_transformed.append(
             (np.where(mls == 0, -1, 1)[i*desired_length:(i+1)*desired_length]*amplitude).tolist())
         print(mls_transformed[i] == mls_transformed[0])
         print(len(mls_transformed[i]))
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"================ handle_mls task, time taken: {elapsed_time}s ================")
    return 200, {
        str(task):{
            "mls": scaled_mls_transformed,
            "unscaledMLS": mls_transformed
        }
    }

def handle_subtracted_psd_task(request_json,task):
    start_time = time.time()
    #print(request_json);
    rec = request_json["rec"]
    # knownGain = request_json["knownGains"]
    # knownFreq = request_json["knownFrequencies"]
    sample_rate = request_json["sampleRate"]
 
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

def handle_ir_convolution_task(request_json, task):
    if "input_signal" not in request_json:
        return 400, "Request Body is missing an 'input_signal' entry"
    if "microphone_ir" not in request_json:
        return 400, "Request Body is missing a 'microphone_ir' entry"
    if "loudspeaker_ir" not in request_json:
        return 400, "Request Body is missing a 'loudspeaker_ir' entry"
    
    input_signal = request_json['input_signal']
    microphone_ir = request_json['microphone_ir']
    loudspeaker_ir = request_json['loudspeaker_ir']
    
    output_signal = run_ir_convolution_task(input_signal, microphone_ir, loudspeaker_ir)
    
    return 200, {
        str(task): {
            'output_signal': output_signal
        }
    }

def handle_frequency_response_task(request_json, task):
    """
    Handler for the frequency response task.
    Converts an impulse response to a frequency response.
    """
    if "impulse_response" not in request_json:
        return 400, "Request Body is missing an 'impulse_response' entry"
    if "sample_rate" not in request_json:
        return 400, "Request Body is missing a 'sample_rate' entry"
    
    impulse_response = request_json["impulse_response"]
    sample_rate = request_json["sample_rate"]
    
    frequencies, gains, gain_at_1000hz = impulse_to_frequency_response(impulse_response, sample_rate)
    
    return 200, {
        str(task): {
            'frequencies': frequencies,
            'gains': gains,
            'gain_at_1000hz': gain_at_1000hz
        }
    }

SUPPORTED_TASKS = {
    'impulse-response': handle_impulse_response_task,
    'autocorrelation': handle_autocorrelation_task,
    'all-hz-check': handle_all_hz_power_check_task,
    'volume-check': handle_volume_power_check_task,
    'component-inverse-impulse-response': handle_component_inverse_impulse_response_task,
    'system-inverse-impulse-response': handle_system_inverse_impulse_response_task,
    'convolution': handle_convolution_task,
    'ir-convolution': handle_ir_convolution_task,
    'volume': handle_volume_task_nonlinear,
    'volume-parameters': handle_volume_parameters,
    'psd': handle_psd_task,
    'subtracted-psd':handle_subtracted_psd_task,
    'mls':handle_mls_task,
    'background-psd': handle_background_psd_task,
    'mls-psd': handle_mls_psd_task,
    'frequency-response': handle_frequency_response_task
}

def print_memory_usage():
    print("memory used:", round(process.memory_info().rss / 1024 ** 2), "mb")

@app.route("/task/<string:task>", methods=['POST'])
@cross_origin()
def task_handler(task):
    print_memory_usage()
    gc.collect()
    if task not in SUPPORTED_TASKS:
        return 'ERROR'
    content_type = request.headers.get('Content-Type')
    if (content_type == 'application/json'):
        headers = {"Content-Type": "application/json"}
        status, result = SUPPORTED_TASKS[task](request.get_json(cache=False), task)
        request.data
        resp = make_response(result, status)
        resp.headers = headers
        print_memory_usage()
        return resp
    else:
        return 'Content-Type not supported'
    
@app.route('/memory', methods=['POST'])
@cross_origin()
def print_memory():
    return {'memory': process.memory_info().rss / 1024 ** 2}    

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
    
@app.route("/model/faceDetect")
@cross_origin()
def face_detect():
    # load and return the JSON File: mediapipe_tfjs_model_face_detection_full.json
    with open('mediapipe_tfjs_model_face_detection_full.json') as f:
        data = json.load(f)
    return data

if __name__ == '__main__':
    app.run()
