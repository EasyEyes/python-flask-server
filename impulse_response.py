import numpy as np
from scipy.fft import fft, ifft, rfft, irfft
from pickle import dumps


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
        # sig = np.roll(sig, -n//2)
        return fft(sig, n=n)


def ifft_sym(sig):
    n = len(sig)
    if n % 2 == 0:
        return irfft(sig.real, n=n)
    else:
        # sig = np.roll(sig, -n//2)
        return ifft(sig, n=n)


def estimate_samples_per_mls_(output_signal, num_periods, sampleRate, L):
    '''
    % 1.1) calculate the autocorrelation of several periods of measured MLS signal

    output_spectrum = fft(output_signal);
    ouptut_autocorrelation = ifft(output_spectrum .* conj(output_spectrum), 'symmetric');
    % ouptut_autocorrelation corresponds to Fig.5 of the paper. 
    '''
    
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
    output_autocorrelation[0:L//2] = 0
    L_new = np.argmax(output_autocorrelation[:output_signal.size//2])
    
    '''
    % find the n-th peak in ouptut_autocorrelation
    % Proceeding a n-th peak increases the precision of estimation
    [~, b] = max(ouptut_autocorrelation(1+n_periods*(L_new-1):n_periods*(L_new+1)));
    L_new_n = b + n_periods*(L_new-1) - 1; % L_new_n is L_new of n_periods
    dL_n = L_new_n - n_periods*L;
    '''
    left = 1*num_periods*(L_new-1)
    right = num_periods*(L_new+1)
    b = np.argmax(output_autocorrelation[left:right])
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
        OUT_MLS2_n = np.zeros(num_periods * L_new_n)
        OUT_MLS2_n[0:cut] = MLS_ADJUST[0:cut].real
    else:
        OUT_MLS2_n = MLS_ADJUST[0:num_periods * L]
    
    return OUT_MLS2_n


def compute_impulse_resp(OUT_MLS2_n, L, fs2):
    '''
    % apply the ifft with Hermitian symmetry
    out_mls2_n = ifft(OUT_MLS2_n, 'symmetric');
    '''
    # N = len(OUT_MLS2_n)
    # right_idx = int(N / 2) + 1
    # out_mls2_n = ifft(OUT_MLS2_n[0:right_idx])
    out_mls2_n = ifft_sym(OUT_MLS2_n)
    
    '''
    % take only the 1st period of MLS to plot the results
    out_mls2 = out_mls2_n(1+L:2*L);
    OUT_MLS2 = fft(out_mls2);
    '''
    out_mls2 = out_mls2_n[1+L:2*L]
    OUT_MLS2 = fft(out_mls2)

    '''
    % correct Impulse Response
    ir = ifft(OUT_MLS2 .* conj(MLS), 'symmetric')./L*2;

    % new frequency axis
    frequency_axis = linspace(0, fs2, length(ir)+1); frequency_axis(end) = [];
    '''
    prod = OUT_MLS2 * OUT_MLS2.conj()
    # N = len(prod)
    # right_idx = int(N / 2) + 1
    # ir = ifft(prod[0:right_idx]) / (L * 2)
    ir = fft_sym(prod) / (L * 2)
    
    return ir

'''
-----------------------------------------------------------------------------------------------------------------------
Tests
'''

def run_ir_task(sig, P=(1 << 18)-1, sampleRate=96000, NUM_PERIODS=2, debug=False):
    sig = np.array(sig, dtype=np.float32)
    
    inpFilt = sig[P+1:]
    
    fs2, L_new_n, dL_n = estimate_samples_per_mls_(inpFilt, NUM_PERIODS, sampleRate, P)
    #print(f'fs2: {fs2}, L_new_n {L_new_n}, dL_n: {dL_n}')
    OUT_MLS2_n = adjust_mls_length(inpFilt, NUM_PERIODS, P, L_new_n, dL_n)
    #print(f'OUT_MLS2_n: {OUT_MLS2_n}')
    ir = compute_impulse_resp(OUT_MLS2_n, P, fs2)

    if debug:
        return ir
    else:
        return str(dumps(ir), encoding="latin1")
    
