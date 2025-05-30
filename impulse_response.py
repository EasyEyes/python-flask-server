import numpy as np
import copy
from scipy.fft import fft, ifft, rfft, irfft, fftfreq
from pickle import dumps
from scipy.interpolate import interp1d


'''
-----------------------------------------------------------------------------------------------------------------------
From Denis
'''   
def zero_pad_pair(a, b):
    len_a = len(a)
    len_b = len(b)
    if(len_a < len_b):
        a_new = np.pad(a, (0, len_b - len_b), 'constant')
        return a_new, b
    else:
        b_new = np.pad(b, (0, len_a - len_b), 'constant')
        return a, b_new

def recover_signal(s, g, h):
    S = fft_sym(s)
    G = fft_sym(g)
    S, G = zero_pad_pair(S, G)
    G_FILTERED = S * G
    H = fft_sym(h)
    G_FILTERED, H = zero_pad_pair(G_FILTERED, H)
    h_filtered = ifft_sym(G_FILTERED * H)
    # g_filtered = convolve(s, g, 'same')
    # h_filtered = convolve(g_filtered, h, 'same')

    return h_filtered

def g_filter(s, g):
    S = fft_sym(s)
    G = fft_sym(g)
    S, G = zero_pad_pair(S, G)
    return ifft_sym(S * G)

'''
-----------------------------------------------------------------------------------------------------------------------
From Novak et al
'''

def fft_sym(sig):
    n = len(sig)
    if n % 2 == 0:
        return rfft(sig.real, n=n)
    else:
        return fft(sig, n=n)


def ifft_sym(sig):
    n = len(sig)
    return irfft(sig,n)[:n]


def estimate_samples_per_mls_(output_signal, num_periods, sampleRate, L):
    '''
    % 1.1) calculate the autocorrelation of several periods of measured MLS signal

    output_spectrum = fft(output_signal);
    ouptut_autocorrelation = ifft(output_spectrum .* conj(output_spectrum), 'symmetric');
    % ouptut_autocorrelation corresponds to Fig.5 of the paper. 
    '''
    if num_periods < 1:
        print("Error, asynchronous algorithm requires at least 2 periods")

    # output_spectrum = np.array(fft(output_signal), dtype=complex)
    output_spectrum = fft(output_signal)
    output_autocorrelation = ifft_sym(output_spectrum * np.conjugate(output_spectrum))
    
    
    # # Find the second-order differences
    # inflection = np.diff(np.sign(np.diff(ouptut_autocorrelation)))
    # peaks = (inflection < 0).nonzero()[0] + 1  # Find where they are negative
    # peaks_idx_sorted = np.argsort(ouptut_autocorrelation[peaks])
    # peak_1_idx = peaks[peaks_idx_sorted[-1]]
    # ouptut_autocorrelation[peak_1_idx] = 0

    '''
    % 1.2) % find the period of ouptut_autocorrelation (locate the second peak)

    ouptut_autocorrelation(1) = 0; % remove the first (maximum) autocorrelation peak
    [~, L_new] = max(ouptut_autocorrelation(1:end/2)); % find the second peak (maximum)
    % L_new is equal to L + dL (Fig. 5)
    % (dL is delta L)
    '''
    corrZero = output_autocorrelation[0]
    output_autocorrelation_1000 = copy.copy(output_autocorrelation[0:1000])
    output_autocorrelation[0:1000] = 0
    L_new = np.argmax(output_autocorrelation[:output_signal.size//2]) + 1
    print("L=" + str(L) + ", L_new=" + str(L_new) + ", based on second peak")
    print("Autocorr at 2nd peak="+str(output_autocorrelation[L_new-1]/corrZero)+", predicted="+str(1-1/num_periods))
    '''
    % find the n-th peak in ouptut_autocorrelation
    % Proceeding a n-th peak increases the precision of estimation
    [~, b] = max(ouptut_autocorrelation(1+n_periods*(L_new-1):n_periods*(L_new+1)));
    L_new_n = b + n_periods*(L_new-1) - 1; % L_new_n is L_new of n_periods
    dL_n = L_new_n - n_periods*L;
    '''
    # num_periods = num_periods-1 
    left = num_periods*(L_new-1)
    right = num_periods*(L_new+1)
    print("Searching for n-th peak in range " + str(left) + " to " + str(right) + " in array of length " + str(len(output_autocorrelation)))
    b = np.argmax(output_autocorrelation[left:right])
    L_new_n = b + num_periods * (L_new - 1)
    dL_n = L_new_n - num_periods * L
    print("Using last peak, recorded " + str(num_periods) + "*MLS period " + str(L_new_n) + " exceeds played " + str(num_periods) + "*MLS period " + str(num_periods*L) + " by fraction " + str(L_new_n/(num_periods*L)-1))
    print("Autocorr at n-th peak="+str(output_autocorrelation[b]/corrZero)+", predicted="+str(1/num_periods))
    
    '''
    % dL_n corresponds to the dL of n-th period of MLS (n = 6 in this example)
    % it is aproximately equal to dL*n_average

    % new sample rate (Eq. (7) of the paper)
    fs2 = fs * L_new_n/(n_periods*L);
    '''
    fs2 = sampleRate * L_new_n / (num_periods * L)
    output_autocorrelation[0:1000] = output_autocorrelation_1000
    return fs2, L_new_n, dL_n, output_autocorrelation


def estimate_samples_per_mls(output_signal, num_periods, sampleRate):
    '''
    % 1.1) calculate the autocorrelation of several periods of measured MLS signal

    output_spectrum = fft(output_signal);
    ouptut_autocorrelation = ifft(output_spectrum .* conj(output_spectrum), 'symmetric');
    % ouptut_autocorrelation corresponds to Fig.5 of the paper. 
    '''
    output_spectrum = fft(output_signal)
    prod = output_spectrum * output_spectrum.conj()
    ouptut_autocorrelation = ifft_sym(prod)
    
    ouptut_autocorrelation[0:10] = 0

    '''
    % 1.2) % find the period of ouptut_autocorrelation (locate the second peak)

    ouptut_autocorrelation(1) = 0; % remove the first (maximum) autocorrelation peak
    [~, L_new] = max(ouptut_autocorrelation(1:end/2)); % find the second peak (maximum)
    % L_new is equal to L + dL (Fig. 5)
    % (dL is delta L)
    '''
    L_new = int(np.max(ouptut_autocorrelation[0:len(ouptut_autocorrelation) // 2]).real)
    
    '''
    % find the n-th peak in ouptut_autocorrelation
    % Proceeding a n-th peak increases the precision of estimation
    [~, b] = max(ouptut_autocorrelation(1+n_periods*(L_new-1):n_periods*(L_new+1)));
    L_new_n = b + n_periods*(L_new-1) - 1; % L_new_n is L_new of n_periods
    dL_n = L_new_n - n_periods*L;
    '''
    left = 1+(num_periods*(L_new - 1))
    right = num_periods*(L_new + 1)
    
    b = int(np.max(ouptut_autocorrelation[left:right]).real)
    
    L_new_n = b + num_periods * (L_new - 1) - 1
    dL_n = L_new_n - num_periods * L_new

    '''
    % dL_n corresponds to the dL of n-th period of MLS (n = 6 in this example)
    % it is aproximately equal to dL*n_average

    % new sample rate (Eq. (7) of the paper)
    fs2 = fs * L_new_n/(n_periods*L);
    '''
    fs2 = sampleRate * L_new_n / (num_periods * L_new)
    
    return fs2, L_new_n, dL_n


def adjust_mls_length(output_signal, num_periods, L, L_new_n, dL_n):
    '''
    MLS_ADJUST = fft(output_signal(1:L_new_n));

    if dL_n < 0 % zero padding (add zeros to the end of the 1st Nyquist zone)

        cut = floor(length(MLS_ADJUST)/2)+1;   % index of end of 1st Nyqyst zone
        OUT_MLS2_n = zeros(n_periods*L,1);     % empty vector
        OUT_MLS2_n(1:cut) = MLS_ADJUST(1:cut); % fill just up to 'cut' (zero padding)
        
    else % remove dL_n points of the spectra
        
        OUT_MLS2_n = MLS_ADJUST(1:n_periods*L);

    end
    '''
    MLS_ADJUST = fft(output_signal[:L_new_n])
    
    if dL_n < 0:
        cut = (len(MLS_ADJUST) // 2) + 1
        OUT_MLS2_n = np.zeros(num_periods * L,dtype=complex)
        OUT_MLS2_n[0:cut] = MLS_ADJUST[0:cut]
    else:
        OUT_MLS2_n = MLS_ADJUST[0:num_periods * L]
    
    return OUT_MLS2_n


def compute_impulse_resp(MLS, OUT_MLS2_n, L, fs2, NUM_PERIODS):
    '''
    % apply the ifft with Hermitian symmetry
    out_mls2_n = ifft(OUT_MLS2_n, 'symmetric');
    '''
    # N = len(OUT_MLS2_n)
    # right_idx = int(N / 2) + 1
    # out_mls2_n = ifft(OUT_MLS2_n[0:right_idx])
    print("Inside compute_impulse_resp")
    print("Length of MLS= " + str(len(MLS)))
    print("Length of OUT_MLS2_n= " + str(len(OUT_MLS2_n)))
    print("L= "+str(L))
    print("fs2= " + str(fs2))
    out_mls2_n = ifft_sym(OUT_MLS2_n)
    print("Length of out_mls2_n= " + str(len(out_mls2_n)))
    '''
    % take only the 1st period of MLS to plot the results
    out_mls2 = out_mls2_n(1+L:2*L);
    OUT_MLS2 = fft(out_mls2);
    '''
    out_mls2 = out_mls2_n[0:NUM_PERIODS*L]
    print('NUM_PERIODS', NUM_PERIODS)
    # out_mls2 = out_mls2_n[0:L]
    print("Length of out_mls2= " + str(len(out_mls2)))
    OUT_MLS2 = fft(out_mls2)
    print("Length of OUT_MLS2= " + str(len(OUT_MLS2)))

    '''
    % correct Impulse Response
    ir = ifft(OUT_MLS2 .* conj(MLS), 'symmetric')./L*2;

    % new frequency axis
    frequency_axis = linspace(0, fs2, length(ir)+1); frequency_axis(end) = [];
    '''
    MLS = np.tile(MLS, NUM_PERIODS)
    MLS = MLS / max(MLS)
    prod = np.multiply(OUT_MLS2, MLS.conj())
    # N = len(prod)
    # right_idx = int(N / 2) + 1
    # ir = ifft(prod[0:right_idx]) / (L * 2)
    # ir = ifft_sym(prod) / L * 2
    ir = ifft_sym(prod) / np.sqrt(len(prod))
    return ir

'''
-----------------------------------------------------------------------------------------------------------------------
Tests
'''

def run_ir_task(mls, sig, P=(1 << 18)-1, sampleRate=96000, NUM_PERIODS=3, debug=False):
    NUM_PERIODS = NUM_PERIODS-1 
    print("number of period ", NUM_PERIODS)
    sig = np.array(sig)
    # mls = list(mls.values())
    mls = np.array(mls)
    MLS = fft(mls)
    L = len(MLS)
    fs2, L_new_n, dL_n, autocorrelation = estimate_samples_per_mls_(sig, NUM_PERIODS, sampleRate, L)
    OUT_MLS2_n = adjust_mls_length(sig, NUM_PERIODS, L, L_new_n, dL_n)
    ir = compute_impulse_resp(MLS, OUT_MLS2_n, L, fs2, NUM_PERIODS)
    if debug:
        return ir, autocorrelation
    else:
        return ir.tolist(), autocorrelation.tolist(), fs2
    

def impulse_to_frequency_response(impulse_response, fs, time_array, total_duration, total_duration_1000hz):
    """
    Convert an impulse response to frequency response (amplitude spectrum).
    
    Args:
        impulse_response: Impulse response signal as numpy array
        fs: Sampling frequency in Hz
        time_array: Array of time values for the impulse response
        total_duration: Total duration for which to pad the impulse response if needed
        total_duration_1000hz:
        
    Returns:
        frequencies: Array of frequency values from 0 to fs/2 with 5 Hz interval
        gains: Array of gain values (amplitude spectrum) corresponding to the frequencies
        gain_at_1000hz: Interpolated gain value at 1000 Hz
        impulse_response: The (potentially padded) impulse response
        impulse_response_1000hz: The impulse response used for 1000 Hz calculation
    """
    # Ensure input is a numpy array
    impulse_response = np.array(impulse_response)
    time_array = np.array(time_array)
    
    # Calculate expected number of samples for the total duration
    expected_samples = int(total_duration/2 * fs)
    expected_samples_1000hz = int(total_duration_1000hz/2 * fs)

    #  create padded array for 1000hz
    impulse_response_1000hz = impulse_response
    if len(impulse_response) < expected_samples_1000hz:
        padded_ir_1000hz = np.zeros(expected_samples_1000hz)
        padded_ir_1000hz[:len(impulse_response)] = impulse_response
        impulse_response_1000hz = padded_ir_1000hz

    # Check if padding is needed
    if len(impulse_response) < expected_samples:
        # Create a padded array filled with zeros
        padded_ir = np.zeros(expected_samples)
        
        # Copy the original impulse response values
        padded_ir[:len(impulse_response)] = impulse_response
        
        # Use the padded impulse response
        impulse_response = padded_ir
    
    # Compute FFT
    fr = fft(impulse_response)
    
    # Get frequency values
    n = len(impulse_response)
    frequencies = fftfreq(n, 1/fs)
    
    # Compute amplitude spectrum (magnitude of frequency response)
    gains = np.abs(fr)
    
    # Keep only positive frequencies and corresponding gains
    pos_mask = frequencies >= 0
    pos_freqs = frequencies[pos_mask]
    pos_gains = gains[pos_mask]
    
    # Create new frequency array from 0 to fs/2 with step of 5 Hz
    nyquist = fs/2
    new_freqs = np.arange(0, nyquist + (5 - nyquist % 5) % 5, 5)
    
    # Create an interpolation function for positive frequencies
    interp_func = interp1d(pos_freqs, pos_gains, kind='linear', bounds_error=False, fill_value="extrapolate")
    
    # Get gains at the new frequency points
    new_gains = interp_func(new_freqs)
    
    # Convert linear gains to dB
    new_gains_db = 20 * np.log10(new_gains)
    
    # Evaluate the interpolation function at 1000 Hz and convert to dB
    gain_at_1000hz_linear = float(interp_func(1000))
    gain_at_1000hz = 20 * np.log10(gain_at_1000hz_linear)
    
    return new_freqs.tolist(), new_gains_db.tolist(), gain_at_1000hz, impulse_response.tolist(), impulse_response_1000hz.tolist()
    
