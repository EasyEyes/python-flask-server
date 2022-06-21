import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq, ifft, rfft, irfft
from scipy import signal
from utils import parser

sampleRate = 96000
N = 18
P = (1 << N) - 1

# ---- Plot Settings ----
PLOT_FONT_SIZE = 10
# plt.style.use(''')
plt.rcParams.update({'font.size': PLOT_FONT_SIZE})


def make_stem(ax, x, y, **kwargs):
    ax.axhline(x[0], x[-1], 0, color='r')

    ax.vlines(x, 0, y, color='b')

    ax.set_ylim([1.05*y.min(), 1.05*y.max()])


def samples_to_miliseconds(num_samples, sample_rate=sampleRate):
    elapsed_time = (num_samples / (sample_rate * 1000)) * 1000
    return np.linspace(0, elapsed_time, num_samples)


def plotPerm(buffer, subsample=False):
    plt.style.use('seaborn')

    if subsample == False:
        subsample = len(buffer)

    max_time = len(buffer) / sampleRate
    time_steps = np.linspace(0, max_time, len(buffer))

    yf = fft(generatedSignal)
    xf = fftfreq(P, 1/sampleRate)

    plt.plot(xf, np.abs(yf), linewidth=1.0)
    plt.show()


def getTimeSteps(signal):
    max_time = len(signal) / sampleRate
    time_steps = np.linspace(0, max_time, len(signal))
    return time_steps

'''
-----------------------------------------------------------------------------------------------------------------------
From Denis
'''   

def compute_filter_g(h, plot=False):
    n = len(h)
    H = fft(h, )
    magnitudes = np.abs(H)
    phases = np.arctan2(H.imag, H.real)

    G_copy = magnitudes * np.exp(1j * phases)
    g_copy = ifft(G_copy)

    G = (1/magnitudes) * np.exp(1j * (-1 * phases))
    G[magnitudes == 0] = 0
    g = ifft(G)
 
    num_to_roll = n//2
    g_rolled = g
    # g_rolled = np.roll(g, num_to_roll)

    if plot:
        fs = 10  # Hz
        t = samples_to_miliseconds(n, fs)

        fig, (ax_H, ax_h, ax_G_copy, ax_g_copy, ax_G,
              ax_g) = plt.subplots(6, 1, figsize=(6.8, 6.8))

        ax_H.plot(t, H, label='H')
        ax_H.set_title('H')
        ax_H.set_xlabel('Time [ms]')
        ax_H.set_ylabel('Amplitude')

        ax_G_copy.plot(t, G_copy, label='G (Copy)')
        ax_G_copy.set_title('G (Copy)')
        ax_G_copy.set_xlabel('Frequency')
        ax_G_copy.set_ylabel('Amplitude')

        ax_G.plot(t, G, label='G')
        ax_G.set_title('G (Inverted)')
        ax_G.set_xlabel('Time [ms]')
        ax_G.set_ylabel('Amplitude')

        ax_h.plot(t, h, label='h')
        ax_h.set_title('h')
        ax_h.set_xlabel('Time [ms]')
        ax_h.set_ylabel('Amplitude')

        ax_g_copy.plot(t, g_copy, label='g')
        ax_g_copy.set_title('g (copy)')
        ax_g_copy.set_xlabel('Time [ms]')
        ax_g_copy.set_ylabel('Amplitude')

        ax_g.plot(t, g_rolled, label='g')
        ax_g.set_title('g (Inverted)')
        ax_g.set_xlabel('Time [ms]')
        ax_g.set_ylabel('Amplitude')

        fig.tight_layout()
        plt.show()

    return g_rolled, G


def compute_filter_g_(h, plot=False):
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

    if plot:
        fs = 10  # Hz
        t = samples_to_miliseconds(n, fs)

        fig, (ax_H, ax_h, ax_G_copy, ax_g_copy, ax_G,
              ax_g) = plt.subplots(6, 1, figsize=(6.8, 6.8))

        ax_H.plot(t, H, label='H')
        ax_H.set_title('H')
        ax_H.set_xlabel('Time [ms]')
        ax_H.set_ylabel('Amplitude')

        ax_G_copy.plot(t, G_copy, label='G (Copy)')
        ax_G_copy.set_title('G (Copy)')
        ax_G_copy.set_xlabel('Frequency')
        ax_G_copy.set_ylabel('Amplitude')

        ax_G.plot(t, G, label='G')
        ax_G.set_title('G (Inverted)')
        ax_G.set_xlabel('Time [ms]')
        ax_G.set_ylabel('Amplitude')

        ax_h.plot(t, h, label='h')
        ax_h.set_title('h')
        ax_h.set_xlabel('Time [ms]')
        ax_h.set_ylabel('Amplitude')

        ax_g_copy.plot(t, g_copy, label='g')
        ax_g_copy.set_title('g (copy)')
        ax_g_copy.set_xlabel('Time [ms]')
        ax_g_copy.set_ylabel('Amplitude')

        ax_g.plot(t, g, label='g')
        ax_g.set_title('g (Inverted)')
        ax_g.set_xlabel('Time [ms]')
        ax_g.set_ylabel('Amplitude')

        fig.tight_layout()
        plt.show()

    return g, G


def recover_signal(s, v, g, h, plot=False):
    g_filtered = signal.convolve(s, g, mode='full')
    h_filtered = signal.convolve(g_filtered, h, mode='full')

    if plot:
        fs = 10  # Hz
        t = samples_to_miliseconds(len(s), fs)

        _, ax_s = plt.subplots()
        s_ss = len(s)
        ax_s.plot(t[:s_ss], s[:s_ss])
        ax_s.set_xlabel(f'Time [ms], {s_ss} samples')
        ax_s.set_ylabel('Amplitude')
        ax_s.set_title('Original Signal (s)')

        _, ax_v = plt.subplots()
        v_ss = len(v)
        ax_v.plot(v[:v_ss])
        ax_v.set_xlabel(f'Time [ms], {v_ss} samples')
        ax_v.set_ylabel('Amplitude')
        ax_v.set_title('Output Signal v (v = h * s), * denotes convolution')

        _, ax_h = plt.subplots()
        h_ss = len(h)
        ax_h.plot( h[:h_ss])
        ax_h.set_xlabel(f'Time [ms], {h_ss} samples')
        ax_h.set_ylabel('Amplitude')
        ax_h.set_title('Impulse Response h (h = ifft( fft(v) / fft(s) ))')

        _, ax_g = plt.subplots()
        g_ss = 10000
        ax_g.plot(g)
        ax_g.set_xlabel(f'Time [ms], {g_ss} samples')
        ax_g.set_ylabel('Amplitude')
        ax_g.set_title('Inverted Impulse Response g')

        _, ax_g_filtered = plt.subplots()
        g_filtered_ss = len(t)
        ax_g_filtered.plot(g_filtered, color='black')
        ax_g_filtered.set_xlabel(f'Time [ms], {g_filtered_ss} samples')
        ax_g_filtered.set_ylabel('Amplitude')
        ax_g_filtered.set_title('Signal sg (sg = g * s), * denotes convolution')

        _, ax_h_filtered = plt.subplots()
        h_filtered_ss = len(t)
        ax_h_filtered.plot(h_filtered, color='black')
        ax_h_filtered.set_xlabel(
            f'Time [ms], {h_filtered_ss} samples')
        ax_h_filtered.set_ylabel('Amplitude')
        ax_h_filtered.set_title(
            'Signal sgh (sgh = g * s * h), * denotes convolution')

        #plot_spectrum(g, fs, t)

        plt.show()

    return h_filtered

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


def estimate_samples_per_mls_(output_signal, num_periods, plot=False):
    '''
    % 1.1) calculate the autocorrelation of several periods of measured MLS signal

    output_spectrum = fft(output_signal);
    ouptut_autocorrelation = ifft(output_spectrum .* conj(output_spectrum), 'symmetric');
    % ouptut_autocorrelation corresponds to Fig.5 of the paper. 
    '''
    output_spectrum = np.array(fft(output_signal), dtype=complex)
    ouptut_autocorrelation = np.array(signal.correlate(output_signal, output_signal, mode='full'), dtype=complex)
    
    # Find the second-order differences
    inflection = np.diff(np.sign(np.diff(ouptut_autocorrelation)))
    peaks = (inflection < 0).nonzero()[0] + 1  # Find where they are negative
    peaks_idx_sorted = np.argsort(ouptut_autocorrelation[peaks])
    peak_1_idx = peaks[peaks_idx_sorted[-1]]
    ouptut_autocorrelation[peak_1_idx] = 0
    
    if plot:
        fig, ax_auto = plt.subplots()
        ax_auto.plot(ouptut_autocorrelation, color='blue')
        ax_auto.plot(peaks, ouptut_autocorrelation[peaks], 'o', color='red')
        ax_auto.plot(peaks[np.argmax(ouptut_autocorrelation[peaks])], np.max(ouptut_autocorrelation[peaks]), 'o', color='yellow')
        plt.show()

    '''
    % 1.2) % find the period of ouptut_autocorrelation (locate the second peak)

    ouptut_autocorrelation(1) = 0; % remove the first (maximum) autocorrelation peak
    [~, L_new] = max(ouptut_autocorrelation(1:end/2)); % find the second peak (maximum)
    % L_new is equal to L + dL (Fig. 5)
    % (dL is delta L)
    '''
    # ouptut_autocorrelation[0] = 0
    L_new = np.argmax(ouptut_autocorrelation[:int(ouptut_autocorrelation.shape[0] / 2)])
    
    '''
    % find the n-th peak in ouptut_autocorrelation
    % Proceeding a n-th peak increases the precision of estimation
    [~, b] = max(ouptut_autocorrelation(1+n_periods*(L_new-1):n_periods*(L_new+1)));
    L_new_n = b + n_periods*(L_new-1) - 1; % L_new_n is L_new of n_periods
    dL_n = L_new_n - n_periods*L;
    '''
    b = np.argmax(ouptut_autocorrelation[int(L_new - 1):int(L_new + 1)])
    L_new_n = b + num_periods * (L_new - 1) - 1
    dL_n = L_new_n - num_periods * L_new

    '''
    % dL_n corresponds to the dL of n-th period of MLS (n = 6 in this example)
    % it is aproximately equal to dL*n_average

    % new sample rate (Eq. (7) of the paper)
    fs2 = fs * L_new_n/(n_periods*L);
    '''
    fs = sampleRate
    fs2 = fs * L_new_n / (num_periods * L_new)
    
    return fs2, L_new_n, dL_n


def estimate_samples_per_mls(output_signal, num_periods, plot=False):
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
    
    if plot:
        _, ax_corr = plt.subplots()
        ax_corr.plot(ouptut_autocorrelation, color='blue')
        ax_corr.plot(output_signal, color='red')
        plt.show()

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
    fs = sampleRate
    fs2 = fs * L_new_n / (num_periods * L_new)
    
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
        OUT_MLS2_n[0:cut] = MLS_ADJUST[0:cut]
    else:
        OUT_MLS2_n = MLS_ADJUST[0:num_periods * L]
    
    return OUT_MLS2_n


def compute_impulse_resp(OUT_MLS2_n, L, fs2, plot=False):
    '''
    % apply the ifft with Hermitian symmetry
    out_mls2_n = ifft(OUT_MLS2_n, 'symmetric');
    '''
    # N = len(OUT_MLS2_n)
    # right_idx = int(N / 2) + 1
    # out_mls2_n = ifft(OUT_MLS2_n[0:right_idx])
    out_mls2_n = fft_sym(OUT_MLS2_n)
    
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
    
    if plot:
        frequency_axis = np.linspace(0, fs2, len(ir))
        _, ax_ir = plt.subplots()
        ax_ir.plot(frequency_axis, ir)
        plt.show()
    
    return ir


'''
-----------------------------------------------------------------------------------------------------------------------
Tests
'''

def run_ir_task(recordedSignals, P=P, sampleRate=96000, NUM_PERIODS=3):
    all_irs = []
    
    for sig in recordedSignals:
        b, a = signal.butter(3, np.array([12e3,20e3])/(sampleRate//2), 'bandpass')
        inpFilt = signal.filtfilt(b, a, sig)
        
        inpFilt = inpFilt[P+1:]
        
        fs2, L_new_n, dL_n = estimate_samples_per_mls_(inpFilt, NUM_PERIODS, plot=False)
        OUT_MLS2_n = adjust_mls_length(inpFilt, NUM_PERIODS, P, L_new_n, dL_n)
        ir = compute_impulse_resp(OUT_MLS2_n, P, fs2, plot=False)
        all_irs.append(ir)
    
    ir = np.mean(all_irs, axis=0)
    g, _ = compute_filter_g(ir, plot=False)
    
    # fig, ax_ir = plt.subplots(1, 1, figsize=(4.8, 4.8))
    # colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'black']
    
    # for i, ir_cur in enumerate(all_irs):
    #     ax_ir.plot(ir_cur, color=colors[i])
        
    # ax_ir.plot(ir, color='black', label='Average')
    # plt.show()
    
    # for sig in recordedSignals:
    #     recovered = recover_signal(generatedSignal, sig, g, ir, plot=True)
    
    return g

if __name__ == '__main__':
    args = parser.parse_args()
    
    # Sample rate of the original signal
    globals()['sampleRate'] = args.sampleRate or 96000
    debug = args.debug  # Print debug info
    prod = not debug
    
    if prod:
        g = run(recordedSignals, P)
        res = ','.join(str(i.real) for i in g)
        
        print(res)
        sys.stdout.flush()  # flush the buffer
    else:
        g = run(recordedSignals, P)
