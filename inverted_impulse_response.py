import numpy as np
import math
from scipy.fft import fft, ifft, irfft, fftfreq
from pickle import loads
from scipy.signal import lfilter, butter, minimum_phase, convolve
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
    return inverse_ir

def calculateInverseIRNoFilter(original_ir, _calibrateSoundIIRPhase, iir_length=500, fs = 96000, componentIRFreqs = None, componentIRGains = None):

    L = iir_length
    # center original IR and prune it to L samples
    nfft = len(original_ir)
    H = np.abs(fft(original_ir))
    ir_new = np.roll(ifft_sym(H),int(nfft/2))
    smoothing_win = 0.5*(1-np.cos(2*np.pi*np.array(range(1,L+1), dtype=np.float32)/(L+1)))
    ir_pruned = ir_new[np.floor(len(ir_new)/2).astype(int)-np.floor(L/2).astype(int):np.floor(len(ir_new)/2).astype(int)+np.floor(L/2).astype(int)] # centered around -l/2 to L/2
    ir_pruned = smoothing_win * ir_pruned

    # calculate inverse from pruned IR, limit to relevant bandwidth and scale
    nfft = L
    H = np.abs(fft(ir_pruned))
        
    iH = np.conj(H)/(np.conj(H)*H)
    if _calibrateSoundIIRPhase == 'minimum':
        iH = np.square(iH)
    inverse_ir = np.roll(ifft_sym(iH),int(nfft/2))
    inverse_ir = scaleInverseResponse(inverse_ir,iH,fs)
    if _calibrateSoundIIRPhase == 'minimum':
        print('calculate inverse impulse response with minimum phase')
        inverse_ir_min = minimum_phase((inverse_ir), method='homomorphic', half=True)
        return inverse_ir_min
    else:
        return inverse_ir
    
def pad_or_truncate_ir(ir, target_length):
    """
    Pad with zeros or truncate an impulse response to match the target length.
    
    Args:
        ir: Input impulse response array
        target_length: Desired length for the output
        
    Returns:
        numpy array: Padded or truncated impulse response
    """
    if len(ir) < target_length:
        # Pad with zeros at the end
        return np.pad(ir, (0, target_length - len(ir)), mode='constant')
    elif len(ir) > target_length:
        # Truncate to target length
        return ir[:target_length]
    else:
        # Already the correct length
        return ir

def frequency_response_to_impulse_response(frequencies, gains, fs, _calibrateSoundIIRPhase, iir_length=500, total_duration=None, total_duration_1000hz=None):
    """
    Convert frequency response to impulse response.
    Similar to calculateInverseIRNoFilter, but starts with frequency data instead of time-domain IR. 

    Args:
        frequencies: Frequencies in Hz
        gains: Gains at each frequency
        fs: Sampling rate

    Returns:
        tuple: (impulse response as numpy array, gain at 1000 Hz)
    """
    # _calibrateSoundIIRPhase = "minimum"
    L_1000hz = int(total_duration_1000hz/2 * fs)
    L_all_hz = int(total_duration/2 * fs)
    # L = max(L_1000hz, L_all_hz)
    L = iir_length

    # Calculate gain at 1000 Hz by interpolation
    gain_at_1000Hz = np.interp(1000, frequencies, gains)

    fft_freqs = np.linspace(0, fs/2, L//2+1)
    
    # Interpolate the gains to match these frequencies
    interpolated_gains = np.interp(fft_freqs, frequencies, gains)
    
    # Create full spectrum (mirror for negative frequencies)
    H = np.concatenate([interpolated_gains, interpolated_gains[-2:0:-1]])

    # change to linear gain: gain = 10^(gain_db/20)
    H = 10**(H/20)

    nfft = len(H)
    ir = np.roll(ifft_sym(H),int(nfft/2))
    if _calibrateSoundIIRPhase == 'minimum':
        print('calculate inverse impulse response with minimum phase')
        ir_min = minimum_phase((ir), method='homomorphic', half=False)
        
        # Use the new padding function for both target lengths
        ir_min_padded = pad_or_truncate_ir(ir_min, L_all_hz)
        ir_min_1000hz_padded = pad_or_truncate_ir(ir_min, L_1000hz)
        
        return ir_min_padded.tolist(), gain_at_1000Hz, fft_freqs.tolist(), interpolated_gains.tolist(), ir_min_1000hz_padded.tolist()
    else:
        # Use the new padding function for both target lengths
        ir_padded = pad_or_truncate_ir(ir, L_all_hz)
        ir_1000hz_padded = pad_or_truncate_ir(ir, L_1000hz)
        
        return ir_padded.tolist(), gain_at_1000Hz, fft_freqs.tolist(), interpolated_gains.tolist(), ir_1000hz_padded.tolist()



def calculateInverseIR(original_ir, lowHz, highHz, _calibrateSoundIIRPhase, iir_length=500, fs = 96000):

    L = iir_length
    # center original IR and prune it to L samples
    nfft = len(original_ir)
    H = np.abs(fft(original_ir))
    ir_new = np.roll(ifft_sym(H),int(nfft/2))
    smoothing_win = 0.5*(1-np.cos(2*np.pi*np.array(range(1,L+1), dtype=np.float32)/(L+1)))
    ir_pruned = ir_new[np.floor(len(ir_new)/2).astype(int)-np.floor(L/2).astype(int):np.floor(len(ir_new)/2).astype(int)+np.floor(L/2).astype(int)] # centered around -l/2 to L/2
    ir_pruned = smoothing_win * ir_pruned

    # calculate inverse from pruned IR, limit to relevant bandwidth and scale
    nfft = L
    H = np.abs(fft(ir_pruned))
    iH = np.conj(H)/(np.conj(H)*H)
    if _calibrateSoundIIRPhase == 'minimum':
        iH = np.square(iH)
    limit_ranges = [lowHz, highHz] #was 100 and 16000
    iH = limitInverseResponseBandwidth(iH, fs, limit_ranges)
    inverse_ir = np.roll(ifft_sym(iH),int(nfft/2))
    #inverse_ir = smoothing_win * inverse_ir
    inverse_ir = scaleInverseResponse(inverse_ir,iH,fs)
    if _calibrateSoundIIRPhase == 'minimum':
        print('calculate inverse impulse response with minimum phase')
        inverse_ir_min = minimum_phase((inverse_ir), method='homomorphic', half=True)
        return inverse_ir_min
    else:
        return inverse_ir
    

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
    print('irLength:', irLength)
    L = irLength
    nfft = len(original_ir)
    H = np.abs(fft(original_ir))
    ir_new = np.roll(ifft_sym(H),int(nfft/2))
    smoothing_win = 0.5*(1-np.cos(2*np.pi*np.array(range(1,L+1), dtype=np.float32)/(L+1)))
    ir_pruned = ir_new[np.floor(len(ir_new)/2).astype(int)-np.floor(L/2).astype(int):np.floor(len(ir_new)/2).astype(int)-np.floor(L/2).astype(int) + L] # centered around -l/2 to L/2
    ir_pruned = smoothing_win * ir_pruned
    return ir_pruned

def smooth_spectrum(spectrum, _calibrateSoundSmoothOctaves=1/3,_calibrateSoundSmoothMinBandwidthHz = 200):
    if _calibrateSoundSmoothOctaves == 0:
        return spectrum
    
    # Compute the ratio r
    r = 2 ** (_calibrateSoundSmoothOctaves / 2)
    print("r", r)
    smoothed_spectrum = np.zeros_like(spectrum)
    
    # Loop through the spectrum and apply smoothing
    for i in range(len(spectrum)):
        # Compute the window indices for averaging
        start_idx = int(max(0, i / r))
        end_idx = int(min(len(spectrum) - 1, i * r))
        bandwidth = (end_idx - start_idx) * 5
        if bandwidth < _calibrateSoundSmoothMinBandwidthHz:
            end_idx = int(min(len(spectrum) - 1, start_idx + _calibrateSoundSmoothMinBandwidthHz / 5))
        
        # Average the points within the window
        smoothed_spectrum[i] = np.mean(spectrum[start_idx:end_idx + 1])
    
    return smoothed_spectrum

def run_component_iir_task(impulse_responses_json, mls, lowHz, highHz, iir_length, componentIRGains,componentIRFreqs,sampleRate, mls_amplitude, irLength, calibrateSoundSmoothOctaves, calibrateSoundSmoothMinBandwidthHz,calibrate_sound_burst_filtered_extra_db, _calibrateSoundIIRPhase, debug=False):
    impulseResponses= impulse_responses_json
    smallest = np.Infinity
    ir = []
    if (len(impulseResponses) > 1):
        for ir in impulseResponses:
            if len(ir) < smallest:
                smallest = len(ir)
        impulseResponses[:] = (ir[:smallest] for ir in impulseResponses)
        ir = np.median(impulseResponses, axis=0)
    else:
        ir = np.array(impulseResponses, dtype=np.float32)
        ir = ir.reshape((ir.shape[1],))
    
    componentIRDeg = np.zeros_like(componentIRFreqs)
    componentIRFreqs = np.array(componentIRFreqs)
    componentIRGains = np.array(componentIRGains)
    ir_component, angle, system_angle = splitter(ir, componentIRFreqs, componentIRGains, componentIRDeg, sampleRate)

    #have my IR here, subtract the microphone/louadspeaker ir from this?
    inverse_response_component = calculateInverseIR(ir_component,lowHz,highHz,_calibrateSoundIIRPhase,iir_length, sampleRate)
    inverse_response_no_bandpass = calculateInverseIRNoFilter(ir_component,_calibrateSoundIIRPhase,iir_length,sampleRate)

    mls = np.array(mls, dtype=np.float32)

    ####cheap transducer trello
    #Convolve three periods of MLS with IIR. Retain only the middle period.
    three_mls_periods = np.tile(mls,3)
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

    frequencies = np.fft.fftfreq(n,d=1/sampleRate)
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
    inverse_response_component = calculateInverseIR(ir_component,lowHz,fMaxHz,_calibrateSoundIIRPhase,iir_length, sampleRate)

    #########
    ir_pruned = prune_ir(ir_component, irLength)
    frequencies = fftfreq(irLength,1/sampleRate)
    ir_fft = fft(ir_pruned)
    component_angle = np.angle(ir_fft,deg=True)
    component_angle = component_angle[:irLength//2]
    return_ir = ir_fft[:len(ir_fft)//2]

    power = abs(return_ir)**2
    power = smooth_spectrum(power, calibrateSoundSmoothOctaves, calibrateSoundSmoothMinBandwidthHz)
    smoothed_return_ir = np.sqrt(power)
    smoothed_return_ir = 20*np.log10(abs(smoothed_return_ir))
    return_ir = 20*np.log10(abs(return_ir))
    return_freq = frequencies[:len(frequencies)//2]
    return inverse_response_component.tolist(), smoothed_return_ir.tolist(), return_freq.real.tolist(),inverse_response_no_bandpass.tolist(), ir_pruned.tolist(), component_angle.tolist(), return_ir.tolist(), system_angle.tolist(), attenuatorGain_dB, fMaxHz

def run_system_iir_task(impulse_responses_json, mls, lowHz, iir_length, highHz, sampleRate, mls_amplitude, calibrate_sound_burst_filtered_extra_db, _calibrateSoundIIRPhase, debug=False):
    impulseResponses= impulse_responses_json
    smallest = np.Infinity
    ir = []
    print('number of impulse response:', len(impulseResponses))
    if (len(impulseResponses) > 1):
        for ir in impulseResponses:
            if len(ir) < smallest:
                smallest = len(ir)
        impulseResponses[:] = (ir[:smallest] for ir in impulseResponses)
        ir = np.median(impulseResponses, axis=0)
    else:
        ir = np.array(impulseResponses)
        ir = ir.reshape((ir.shape[1],))
    inverse_response= calculateInverseIR(ir,lowHz,highHz, _calibrateSoundIIRPhase,iir_length,sampleRate)
    inverse_response_no_bandpass = calculateInverseIRNoFilter(ir,_calibrateSoundIIRPhase, iir_length,sampleRate)

    mls = np.array(mls)
    ####cheap transducer trello
    #Convolve three periods of MLS with IIR. Retain only the middle period.
    three_mls_periods = np.tile(mls,3)
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

    frequencies = np.fft.fftfreq(n,d=1/sampleRate)
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
    inverse_response= calculateInverseIR(ir,lowHz,fMaxHz,_calibrateSoundIIRPhase,iir_length, sampleRate)

    return inverse_response.tolist(), ir.real.tolist(), inverse_response_no_bandpass.tolist(), attenuatorGain_dB, fMaxHz

def run_convolution_task(inverse_response, mls, inverse_response_no_bandpass, attenuatorGain_dB, mls_amplitude):
     
    orig_mls = mls
    N = 1 + math.ceil(len(inverse_response)/len(mls))
    print('N: ' + str(N))
    mls = np.tile(mls,N)
    print('length of tiled mls: ' + str(len(mls)))
    print('length of inverse_response: ' + str(len(inverse_response)))
    convolution = lfilter(inverse_response,1,mls)
    convolution_no_bandpass = lfilter(inverse_response_no_bandpass,1,mls)

    print('length of original convolution: ' + str(len(convolution)))
    trimmed_convolution = convolution[(len(orig_mls)*(N-1)):]
    trimmed_convolution_no_bandpass = convolution_no_bandpass[(len(orig_mls)*(N-1)):]
    convolution_div = trimmed_convolution * mls_amplitude #really amplitude
    convolution_div_no_bandpass = trimmed_convolution_no_bandpass * mls_amplitude
    print("ATTENUATION gain: ", attenuatorGain_dB)
    if (attenuatorGain_dB != 0):
        convolution_div = convolution_div * (10**(attenuatorGain_dB/20))
        convolution_div_no_bandpass = convolution_div_no_bandpass * (10**(attenuatorGain_dB/20))
    print('length of convolution: ' + str(len(trimmed_convolution)))

    maximum = max(convolution_div)
    minimum = min(convolution_div)
    print("Max value convolution: " + str(maximum))
    print("Min value convolution: " + str(minimum))
    return convolution_div.tolist(), convolution_div_no_bandpass.tolist()

def run_ir_convolution_task(input_signal, microphone_ir, loudspeaker_ir, sample_rate, duration):
    """
    Convolve an input signal with both microphone and loudspeaker impulse responses.
    
    Args:
        input_signal: Input audio signal as numpy array
        microphone_ir: Microphone impulse response as numpy array
        loudspeaker_ir: Loudspeaker impulse response as numpy array
        sample_rate: Sample rate of the input signal
        duration: duration needed for the input signal. Repeats the input signal if necessary.
        
    Returns:
        The convolved output signal as a list
    """
    # Convert to numpy arrays if needed
    input_signal = np.array(input_signal)
    microphone_ir = np.array(microphone_ir)
    loudspeaker_ir = np.array(loudspeaker_ir)

    length_of_input_signal = len(input_signal)
    required_length = int(sample_rate * duration)
    # repeat the input signal if necessary so that it is the same length as the required length
    if length_of_input_signal < required_length:
        input_signal = np.tile(input_signal, math.ceil(required_length / length_of_input_signal))

    
    # For efficiency, convolve with the shorter IR first
    if len(microphone_ir) <= len(loudspeaker_ir):
        # First convolve with microphone IR
        intermediate_signal = convolve(input_signal, microphone_ir, mode='full')
        # Then convolve with loudspeaker IR
        output_signal = convolve(intermediate_signal, loudspeaker_ir, mode='full')
    else:
        # First convolve with loudspeaker IR
        intermediate_signal = convolve(input_signal, loudspeaker_ir, mode='full')
        # Then convolve with microphone IR
        output_signal = convolve(intermediate_signal, microphone_ir, mode='full')

    # make output signal same length as the required length
    output_signal = output_signal[:required_length]
    
    return output_signal.tolist()
