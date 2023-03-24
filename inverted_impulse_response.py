import numpy as np
from scipy.fft import fft, ifft, irfft
from pickle import loads
from scipy.signal import lfilter

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
    # set gain of freqs below and above the limits to 1. Note it's a two sided spectrum
    for i,freq in enumerate(frequencies):
        cond1 = (freq < limit_ranges[0]) or (freq > frequencies[-1] - limit_ranges[0])
        cond2 = (freq > limit_ranges[1]) and (freq < frequencies[-1] - limit_ranges[1])
        if cond1 or cond2:
            inverse_spectrum[i] = 1.

    return inverse_spectrum

def scaleInverseResponse(inverse_ir, inverse_spectrum, fs, target=1000):
    frequencies = np.linspace(0,fs,len(inverse_spectrum)+1)[:-1]
    freq_target_idx = (np.abs(frequencies - target)).argmin()
    scale_value = inverse_spectrum[freq_target_idx]
    inverse_ir = inverse_ir/scale_value
    return inverse_ir, scale_value

def calculateInverseIR(original_ir, lowHz, highHz, L=500, fs = 96000):

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

def run_iir_task(impulse_responses_json, mls, lowHz, highHz, debug=False):
    impulseResponses= impulse_responses_json
    smallest = np.Infinity
    for ir in impulseResponses:
        if len(ir) < smallest:
            smallest = len(ir)
    impulseResponses[:] = (ir[:smallest] for ir in impulseResponses)
    ir = np.mean(impulseResponses, axis=0)

    inverse_response, scale, ir_pruned = calculateInverseIR(ir,lowHz,highHz)
    mls = list(mls.values())
    mls = np.array(mls)
    mls_pad = np.pad(mls, (0, 500), 'constant')
    convolution = lfilter(inverse_response,1,mls_pad)
    maximum = max(convolution)
    minimum = abs(min(convolution))
    divisor = 0
    if maximum > minimum:
        divisor = maximum
    else:
        divisor = minimum

    convolution_div = convolution/divisor
    convolution_div = convolution_div*.1

    return inverse_response.tolist(), convolution_div.tolist()