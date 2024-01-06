import numpy as np
import math
from scipy.fft import fft, ifft, irfft, fftfreq
from pickle import loads
from scipy.signal import lfilter, butter
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

def scaleInverseResponse(inverse_ir, inverse_spectrum, fs, targetHz=1000):

    # Old method, using both inverse_ir and inverse_spectrum.
    frequencies = np.linspace(0,fs,len(inverse_spectrum)+1)[:-1]
    freq_target_idx = (np.abs(frequencies - targetHz)).argmin()
    scale_value = inverse_spectrum[freq_target_idx]
    print('Using both inverse_ir and inverse_spectrum: targetHz ' + str(frequencies[freq_target_idx]) + ", scale_value = "+ str(scale_value))

    # New method, using only inverse_ir.
    # Use frequency closest to requested targetHz that has an integer  
    # number of periods in inverse_ir.
    targetHz = round(targetHz * len(inverse_ir)/fs) / (len(inverse_ir)/fs)
    ii = np.arange(0,len(inverse_ir))
    radians = 2 * np.pi * targetHz * ii / fs
    a = np.sum(inverse_ir * np.sin(radians))
    b = np.sum(inverse_ir * np.cos(radians))
    scale_value = np.sqrt(a**2 + b**2)
    print(f'Using only inverse_ir: targetHz {targetHz:.3f}; scale_value={scale_value:.3f}')

    inverse_ir = inverse_ir/scale_value
    return inverse_ir, scale_value

def calculateInverseIRNoFilter(original_ir, iir_length=500, fs = 96000, componentIRFreqs = None, componentIRGains = None):

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
    #inverse_ir = smoothing_win * inverse_ir
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
    #inverse_ir = smoothing_win * inverse_ir
    inverse_ir, scale_value = scaleInverseResponse(inverse_ir,iH,fs)

    return inverse_ir, scale_value, ir_pruned

def splitter(system_ir,partIRHz,partIRDb,partIRDeg,fs=48000):
  systemSpectrum = fft(system_ir)
  systemGain = np.abs(systemSpectrum)
  systemDeg = np.angle(systemSpectrum,deg=True) # radians → deg
  num_samples = len(systemGain)
  frequenciesHz = fftfreq(num_samples,1/fs)
  print("frequencies increasing", np.all(np.diff(partIRHz) > 0))
  # linearly interpolate gain and phase
  partDb=np.interp(frequenciesHz,partIRHz,partIRDb)
  partDeg=np.interp(frequenciesHz,partIRHz,partIRDeg)
  otherGain=systemGain/10**(partDb/20)
  otherDeg=systemDeg-partDeg

  otherSpectrum = otherGain*np.exp(1j*np.deg2rad(otherDeg))
  n=int(len(system_ir)/2)
  other_ir=np.roll(ifft_sym(otherSpectrum),n)
  systemDeg = systemDeg[:num_samples//2]
  otherDeg = otherDeg[:num_samples//2]
  return other_ir, otherDeg, systemDeg

def prune_ir(original_ir, irLength):
    L = irLength
    nfft = len(original_ir)
    H = np.abs(fft(original_ir))
    ir_new = np.roll(ifft_sym(H),int(nfft/2))
    smoothing_win = 0.5*(1-np.cos(2*np.pi*np.array(range(1,L+1))/(L+1)))
    ir_pruned = ir_new[np.floor(len(ir_new)/2).astype(int)-np.floor(L/2).astype(int):np.floor(len(ir_new)/2).astype(int)-np.floor(L/2).astype(int) + L] # centered around -l/2 to L/2
    ir_pruned = smoothing_win * ir_pruned
    return ir_pruned

def smooth_spectrum(spectrum, _calibrateSoundSmoothOctaves=1/3):
    if _calibrateSoundSmoothOctaves == 0:
        return spectrum
    
    # Compute the ratio r
    r = 2 ** (_calibrateSoundSmoothOctaves / 2)
    
    smoothed_spectrum = np.zeros_like(spectrum)
    
    # Loop through the spectrum and apply smoothing
    for i in range(len(spectrum)):
        # Compute the window indices for averaging
        start_idx = int(max(0, i / r))
        end_idx = int(min(len(spectrum) - 1, i * r))
        
        # Average the points within the window
        smoothed_spectrum[i] = np.mean(spectrum[start_idx:end_idx + 1])
    
    return smoothed_spectrum

def run_component_iir_task(impulse_responses_json, mls, lowHz, highHz, iir_length, componentIRGains,componentIRFreqs,num_periods,sampleRate, mls_amplitude, irLength, calibrateSoundSmoothOctaves, calibrate_sound_burst_filtered_extra_db, debug=False):
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

    sample_rate = sampleRate
    
    componentIRDeg = np.zeros_like(componentIRFreqs)
    ir_component, angle, system_angle = splitter(ir, componentIRFreqs, componentIRGains, componentIRDeg, sample_rate)

    #have my IR here, subtract the microphone/louadspeaker ir from this?
    inverse_response_component, scale, _ = calculateInverseIR(ir_component,lowHz,highHz,iir_length, sample_rate)
    inverse_response_no_bandpass, _, _ = calculateInverseIRNoFilter(ir_component,iir_length,sample_rate)

    mls = np.array(mls)
    orig_mls = mls

    ####cheap transducer trello
    #Convolve three periods of MLS with IIR. Retain only the middle period.
    three_mls_periods = np.tile(mls,3);
    three_mls_periods_convolution = lfilter(inverse_response_component,1,three_mls_periods)
    period_length = len(mls)
    start_index = period_length
    end_index = start_index + period_length
    middle_period_convolution = three_mls_periods_convolution[start_index:end_index]
    middle_period_convolution = middle_period_convolution * mls_amplitude
    #middle_period_convolution = mls * mls_amplitude
    #compute fft and cumulative power below the cut of frequency as a function of the cut off frequency
    fft_result = np.fft.fft(middle_period_convolution)
    fft_magnitude = np.abs(fft_result)
    half_spectrum = fft_magnitude[:len(fft_result) // 2]
    n = len(middle_period_convolution)

    power_spectrum = fft_magnitude**2

    frequencies = np.fft.fftfreq(n,d=1/sample_rate)
    frequencies = frequencies[:len(frequencies) // 2]

    pcum = np.cumsum(half_spectrum)
    total_power = np.mean(middle_period_convolution**2)
    pcum = total_power*pcum/pcum[-1]
    # If MLSPower < PCum(inf) then set fMaxHz to the cut off frequency at which integrated power is MLSPower. 
    #In MATLAB I would use the interpolation function interp1. Most languages have a similar interpolation function.
    pcum_infinity = pcum[-1]

    mls_power = mls_amplitude ** 2
    mls_power_db = 10*np.log10(mls_power)
    fMaxHz = 0
    attenuatorGain_dB = 0
    #print outs
    print('calibrate_sound_burst_filtered_extra_db ' + str(calibrate_sound_burst_filtered_extra_db))
    calibrate_sound_burst_filtered_power_factor = 10 ** ( calibrate_sound_burst_filtered_extra_db / 10)

    print('mls_power_db {:.1f}'.format(mls_power_db))
    print('pcum[-1]  {:.1f} dB'.format(10*np.log10(pcum[-1])))
    print('Min frequency: {:.0f} Hz'.format(min(frequencies)))
    print('Max frequency: {:.0f} Hz'.format(max(frequencies)))
    for i in range(0, len(frequencies), round(len(frequencies)/10)):
        print(round(frequencies[i]), end=' ')

    power_limit = mls_power*calibrate_sound_burst_filtered_power_factor
    if (power_limit < pcum_infinity):
        fMaxHz = np.interp(power_limit, pcum, frequencies)
        fMaxHz = round(fMaxHz /100) * 100
        print("power_limit < pcum_infinity")
        print('fMaxHz {:.0f} Hz'.format(fMaxHz))
        if (fMaxHz > 1500):
            attenuatorGain_dB = 0
            print('fmax > 1500')
            print('fMaxHz {:.0f} Hz'.format(fMaxHz))
            fMaxHz = min(fMaxHz, highHz)
        else:
            fMaxHz = 1500
            pcum_1500 = np.interp(1500, frequencies, pcum)
            attenuatorGain_dB = mls_power_db - 10*np.log10(pcum_1500)
    else:
        print("power_limit > pcum_infinity")
        fMaxHz = highHz
        attenuatorGain_dB = 0


    ####apply lowpass filter
    inverse_response_component, _, _ = calculateInverseIR(ir_component,lowHz,fMaxHz,iir_length, sample_rate)

    #########
    N = 1 + math.ceil(len(inverse_response_component)/len(mls))
    print('N: ' + str(N))
    mls = np.tile(mls,N)
    print('length of tiled mls: ' + str(len(mls)))
    print('length of inverse_response: ' + str(len(inverse_response_component)))
    convolution = lfilter(inverse_response_component,1,mls)
    print('length of original convolution: ' + str(len(convolution)))

    trimmed_convolution = convolution[(len(orig_mls)*(N-1)):]
    convolution_div = trimmed_convolution * mls_amplitude
    print("ATTENUATION gain")
    print(attenuatorGain_dB)
    print("fMaxHz")
    print(fMaxHz)
    if (attenuatorGain_dB != 0):
        convolution_div = convolution_div * (10**(attenuatorGain_dB/20))
    print('length of convolution: ' + str(len(trimmed_convolution)))
    print(len(trimmed_convolution))


    maximum = max(convolution_div)
    minimum = min(convolution_div)
    print("Max value component convolution: " + str(maximum))
    print("Min value component convolution: " + str(minimum))

    ir_pruned = prune_ir(ir_component, irLength)
    num_samples = len(ir_pruned)
    frequencies = fftfreq(num_samples,1/sample_rate)
    ir_fft = fft(ir_pruned)
    component_angle = np.angle(ir_fft,deg=True)
    component_angle = component_angle[:num_samples//2]
    #angle = angle[:len(angle)//2]
    #system_angle = system_angle[:len(system_angle)//2]
    return_ir = ir_fft[:len(ir_fft)//2]
    ## DELETE: return_ir = 20*np.log10(abs(return_ir))
    power = abs(return_ir)**2
    power = smooth_spectrum(power, calibrateSoundSmoothOctaves)
    smoothed_return_ir = np.sqrt(power)
    smoothed_return_ir = 20*np.log10(abs(smoothed_return_ir))
    return_ir = 20*np.log10(abs(return_ir))
    return_freq = frequencies[:len(frequencies)//2]
    return inverse_response_component.tolist(), convolution_div.tolist(), smoothed_return_ir.tolist(), return_freq.real.tolist(),inverse_response_no_bandpass.tolist(), ir_component.tolist(), component_angle.tolist(), return_ir.tolist(), system_angle.tolist(), attenuatorGain_dB, fMaxHz

def run_system_iir_task(impulse_responses_json, mls, lowHz, iir_length, highHz, num_periods, sampleRate, mls_amplitude, calibrate_sound_burst_filtered_extra_db, debug=False):
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
    # mls = list(mls.values())
    mls = np.array(mls)
    orig_mls = mls
    ####cheap transducer trello
    #Convolve three periods of MLS with IIR. Retain only the middle period.
    sample_rate=sampleRate
    three_mls_periods = np.tile(mls,3);
    three_mls_periods_convolution = lfilter(inverse_response,1,three_mls_periods)
    period_length = len(mls)
    start_index = period_length
    end_index = start_index + period_length
    middle_period_convolution = three_mls_periods_convolution[start_index:end_index]
    middle_period_convolution = middle_period_convolution * mls_amplitude
    #middle_period_convolution = mls * mls_amplitude
    #compute fft and cumulative power below the cut of frequency as a function of the cut off frequency
    fft_result = np.fft.fft(middle_period_convolution)
    fft_magnitude = np.abs(fft_result)
    half_spectrum = fft_magnitude[:len(fft_result) // 2]
    n = len(middle_period_convolution)

    power_spectrum = fft_magnitude**2

    frequencies = np.fft.fftfreq(n,d=1/sample_rate)
    frequencies = frequencies[:len(frequencies) // 2]

    pcum = np.cumsum(half_spectrum)
    total_power = np.mean(middle_period_convolution**2)
    pcum = total_power*pcum/pcum[-1]
    # If MLSPower < PCum(inf) then set fMaxHz to the cut off frequency at which integrated power is MLSPower. 
    #In MATLAB I would use the interpolation function interp1. Most languages have a similar interpolation function.
    pcum_infinity = pcum[-1]

    mls_power = mls_amplitude ** 2
    mls_power_db = 10*np.log10(mls_power)
    fMaxHz = 0
    attenuatorGain_dB = 0
    #print outs
    print('calibrate_sound_burst_filtered_extra_db ' + str(calibrate_sound_burst_filtered_extra_db))
    calibrate_sound_burst_filtered_power_factor = 10 ** ( calibrate_sound_burst_filtered_extra_db / 10)

    print('mls_power_db {:.1f}'.format(mls_power_db))
    print('pcum[-1]  {:.1f} dB'.format(10*np.log10(pcum[-1])))
    print('Min frequency: {:.0f} Hz'.format(min(frequencies)))
    print('Max frequency: {:.0f} Hz'.format(max(frequencies)))

    for i in range(0, len(frequencies), round(len(frequencies)/10)):
        print(round(frequencies[i]), end=' ')
    power_limit = mls_power*calibrate_sound_burst_filtered_power_factor
    if (power_limit < pcum_infinity):
        fMaxHz = np.interp(power_limit, pcum, frequencies)
        fMaxHz = round(fMaxHz /100) * 100
        if (fMaxHz > 1500):
            attenuatorGain_dB = 0
            fMaxHz = min(fMaxHz, highHz)
        else:
            fMaxHz = 1500
            pcum_1500 = np.interp(1500, frequencies, pcum)
            print("PCUM 1500")
            print(pcum_1500)
            print("MLS POWER DB")
            print(mls_power_db)
            attenuatorGain_dB = mls_power_db - 10*np.log10(pcum_1500)
    else:
        fMaxHz = highHz
        attenuatorGain_dB = 0


    ####apply lowpass filter
    inverse_response, _, _ = calculateInverseIR(ir,lowHz,fMaxHz,iir_length, sample_rate)

    ######new method
    N = 1 + math.ceil(len(inverse_response)/len(mls))
    print('N: ' + str(N))
    mls = np.tile(mls,N)
    print('length of tiled mls: ' + str(len(mls)))
    print('length of inverse_response: ' + str(len(inverse_response)))
    convolution = lfilter(inverse_response,1,mls)
    #convolution = np.convolve(inverse_response,mls)
    #print('length of original convolution: ' + str(len(convolution)))
   # start_index = (N-1)*len(orig_mls)
    #print('start index: ' + str(start_index))
    #end_index = -len(inverse_response)
    #print('end index: ' + str(end_index))
    #trimmed_convolution = convolution[start_index:end_index]
    print('length of original convolution: ' + str(len(convolution)))
    trimmed_convolution = convolution[(len(orig_mls)*(N-1)):]
    convolution_div = trimmed_convolution * mls_amplitude #really amplitude
    print("ATTENUATION gain")
    print(attenuatorGain_dB)
    print("fMaxHz")
    print(fMaxHz)
    if (attenuatorGain_dB != 0):
        convolution_div = convolution_div * (10**(attenuatorGain_dB/20))
    print('length of convolution: ' + str(len(trimmed_convolution)))
    print(len(trimmed_convolution))

    maximum = max(convolution_div)
    minimum = min(convolution_div)
    print("Max value convolution: " + str(maximum))
    print("Min value convolution: " + str(minimum))
    #########

    ##########old method 
    # mls= np.tile(mls, num_periods)
    # mls_pad = np.pad(mls, (0, iir_length), 'constant')
    # convolution = lfilter(inverse_response,1,mls_pad)

    # print("Max convolution")
    # maximum = max(convolution)
    # print(maximum)
    # minimum = abs(min(convolution))
    # print("Min convolution")
    # print(minimum)
    # print("Root mean squared of convolution")
    # rms = np.sqrt(np.mean(np.square(convolution)))
    # print(rms)
    # divisor = 0
    # if maximum > minimum:
    #     divisor = maximum
    # else:
    #     divisor = minimum

    # convolution_div = convolution
    #############

    return inverse_response.tolist(), convolution_div.tolist(), ir.real.tolist(), inverse_response_no_bandpass.tolist(), attenuatorGain_dB, fMaxHz