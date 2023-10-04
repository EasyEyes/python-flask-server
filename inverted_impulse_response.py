import numpy as np
from scipy.fft import fft, ifft, irfft, fftfreq
from pickle import loads
from scipy.signal import lfilter
from scipy.interpolate import interp1d

def ifft_sym(sig):
    n = len(sig)
    return irfft(sig,n)[:n]

def compute_filter_g_(h):
    H = fft(h)
    n = len(h)

    C = np.log(np.abs(H))
    c = ifft(C, n=n)

    m = np.empty(n, dtype=complex)
    m[0] = c[0]
    m[n//2] = c[n//2]
    m[1:n//2] = 2 * c[1:n//2]
    m[n//2:] = 0

    M = fft(m, n=n)
    Mk = np.exp(M)

    G = 1/Mk
    g = ifft(G, n=n)

    G_copy = H
    g_copy = h

    return g, G

def compute_filter_g(h):
    H = fft(h)
    magnitudes = np.abs(H)
    phases = np.arctan2(H.imag, H.real)

    G = (1/magnitudes) * np.exp(1j * (-1 * phases))
    G[magnitudes == 0] = 0
    g = ifft(G).real
 
    return g


def limitInverseResponseBandwidth(inverse_spectrum, fs, limit_ranges):
    frequencies = np.linspace(0,fs,len(inverse_spectrum)+1)[:-1]
    # set gain of freqs below and above the limits to 0. Note it's a two sided spectrum
    for i,freq in enumerate(frequencies):
        cond1 = (freq < limit_ranges[0]) or (freq > frequencies[-1] - limit_ranges[0])
        cond2 = (freq > limit_ranges[1]) and (freq < frequencies[-1] - limit_ranges[1])
        if cond1 or cond2:
            inverse_spectrum[i] = 0.

    return inverse_spectrum #add inverse_spectrum

def scaleInverseResponse(inverse_ir, inverse_spectrum, fs, target=1000):
    frequencies = np.linspace(0,fs,len(inverse_spectrum)+1)[:-1]
    freq_target_idx = (np.abs(frequencies - target)).argmin()
    scale_value = inverse_spectrum[freq_target_idx]
    inverse_ir = inverse_ir/scale_value
    return inverse_ir, scale_value

def calculateInverseIRNoFilter(original_ir, iir_length=500, fs = 96000):

    L = iir_length
    # center original IR and prune it to L samples
    nfft = len(original_ir)
    H = np.abs(fft(original_ir))
    ir_new = np.roll(ifft_sym(H),int(nfft/2))
    smoothing_win = 0.5*(1-np.cos(2*np.pi*np.array(range(1,L+1))/(L+1)))
    ir_pruned = ir_new[np.floor(len(ir_new)/2).astype(int)-np.floor(L/2).astype(int):np.floor(len(ir_new)/2).astype(int)+np.floor(L/2).astype(int)] # centered around -l/2 to L/2
    ir_pruned = smoothing_win * ir_pruned

    # calculate inverse from pruned IR, limit to relevant bandwidth and scale
    nfft = L
    H = np.abs(fft(ir_pruned))
    iH = np.conj(H)/(np.conj(H)*H)
    inverse_ir = np.roll(ifft_sym(iH),int(nfft/2))
    inverse_ir = smoothing_win * inverse_ir
    inverse_ir, scale_value = scaleInverseResponse(inverse_ir,iH,fs)

    return inverse_ir, scale_value, ir_pruned

def calculateInverseIR(original_ir, lowHz, highHz, iir_length=500, fs = 96000):

    L = iir_length
    # center original IR and prune it to L samples
    nfft = len(original_ir)
    H = np.abs(fft(original_ir))
    ir_new = np.roll(ifft_sym(H),int(nfft/2))
    smoothing_win = 0.5*(1-np.cos(2*np.pi*np.array(range(1,L+1))/(L+1)))
    ir_pruned = ir_new[np.floor(len(ir_new)/2).astype(int)-np.floor(L/2).astype(int):np.floor(len(ir_new)/2).astype(int)+np.floor(L/2).astype(int)] # centered around -l/2 to L/2
    ir_pruned = smoothing_win * ir_pruned

    # calculate inverse from pruned IR, limit to relevant bandwidth and scale
    nfft = L
    H = np.abs(fft(ir_pruned))
    iH = np.conj(H)/(np.conj(H)*H)
    limit_ranges = [lowHz, highHz] #was 100 and 16000
    iH = limitInverseResponseBandwidth(iH, fs, limit_ranges)
    inverse_ir = np.roll(ifft_sym(iH),int(nfft/2))
    inverse_ir = smoothing_win * inverse_ir
    inverse_ir, scale_value = scaleInverseResponse(inverse_ir,iH,fs)

    return inverse_ir, scale_value, ir_pruned

def run_component_iir_task(impulse_responses_json, mls, lowHz, highHz, iir_length, componentIRGains,componentIRFreqs,num_periods,sampleRate,debug=False):
    impulseResponses= impulse_responses_json
    smallest = np.Infinity
    ir = []
    if (len(impulseResponses) > 1):
        for ir in impulseResponses:
            if len(ir) < smallest:
                smallest = len(ir)
        impulseResponses[:] = (ir[:smallest] for ir in impulseResponses)
        ir = np.mean(impulseResponses, axis=0) #time domain
    else:
        ir = np.array(impulseResponses)
        ir = ir.reshape((ir.shape[1],))
    ir_fft = fft(ir)
    sample_rate = sampleRate
    num_samples = len(ir)
    frequencies = fftfreq(num_samples,1/sample_rate)
    
    #interpolation part
    #1) convert ir_fft to dB
    ir_fft_db = 20*np.log10(abs(ir_fft))
    #2) interpolate and subtract
    #interpolate function for componentGains and componentFreqs
    interp_func = interp1d(componentIRFreqs,componentIRGains)

    #3) some sorting to make sure gains and frequencies are in sorted order when returned
    min_freq = min(componentIRFreqs)
    max_freq = max(componentIRFreqs)
    inbounds_indices = np.where((abs(frequencies) >= min_freq) & (abs(frequencies) <= max_freq))
    outbounds_indices = np.where((abs(frequencies) <= min_freq) & (abs(frequencies) >= max_freq))
    inbounds_frequencies = abs(frequencies[inbounds_indices])
    inbounds_ir_fft_db = ir_fft_db[inbounds_indices]
    interp_gain2 = interp_func(inbounds_frequencies)
    result = inbounds_ir_fft_db - interp_gain2
    final_result = np.zeros_like(frequencies)
    final_result[inbounds_indices] = result
    final_result[outbounds_indices] = ir_fft_db[outbounds_indices]
 
    #3) convert ir_fft to linear, invert back to time
    ir = 10**(final_result/20)
    ir_fft = ir
    #ir = ifft(ir)
    nfft = len(ir)
    ir = np.roll(ifft_sym(ir),int(nfft/2))

    
    #have my IR here, subtract the microphone/louadspeaker ir from this?
    inverse_response, scale, ir_pruned = calculateInverseIR(ir,lowHz,highHz,iir_length, sample_rate)
    inverse_response_no_bandpass, _, _ = calculateInverseIRNoFilter(ir,iir_length,sample_rate)
    mls = list(mls.values())
    mls = np.array(mls)
    mls= np.tile(mls, num_periods)
    mls_pad = np.pad(mls, (0, iir_length), 'constant')
    convolution = lfilter(inverse_response,1,mls_pad)
    print("Max convolution value")
    print(max(convolution)
    print("Min convolution value")
    print(min(convolution))
    # maximum = max(convolution)
    # minimum = abs(min(convolution))
    # divisor = 0
    # if maximum > minimum:
    #     divisor = maximum
    # else:
    #     divisor = minimum

    convolution_div = convolution
    return_ir = ir_fft[:len(ir_fft)//2]
    return_freq = frequencies[:len(frequencies)//2]

    return inverse_response.tolist(), convolution_div.tolist(), return_ir.real.tolist(), return_freq.real.tolist(),inverse_response_no_bandpass.tolist()

def run_system_iir_task(impulse_responses_json, mls, lowHz, iir_length, highHz,num_periods,sampleRate,debug=False):
    impulseResponses= impulse_responses_json
    smallest = np.Infinity
    ir = []
    if (len(impulseResponses) > 1):
        for ir in impulseResponses:
            if len(ir) < smallest:
                smallest = len(ir)
        impulseResponses[:] = (ir[:smallest] for ir in impulseResponses)
        ir = np.mean(impulseResponses, axis=0)
    else:
        ir = np.array(impulseResponses)
        ir = ir.reshape((ir.shape[1],))
    inverse_response, scale, ir_pruned = calculateInverseIR(ir,lowHz,highHz, iir_length,sampleRate)
    inverse_response_no_bandpass, _, _ = calculateInverseIRNoFilter(ir,iir_length,sampleRate)
    mls = list(mls.values())
    mls = np.array(mls)
    mls= np.tile(mls, num_periods)
    mls_pad = np.pad(mls, (0, iir_length), 'constant')
    convolution = lfilter(inverse_response,1,mls_pad)
    print("Max convolution value")
    print(max(convolution)
    print("Min convolution value")
    print(min(convolution))
    # maximum = max(convolution)
    # minimum = abs(min(convolution))
    # divisor = 0
    # if maximum > minimum:
    #     divisor = maximum
    # else:
    #     divisor = minimum

    convolution_div = convolution

    return inverse_response.tolist(), convolution_div.tolist(), ir.real.tolist(), inverse_response_no_bandpass.tolist()