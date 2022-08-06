import numpy as np
from scipy.fft import fft, ifft
from pickle import loads


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
 
    return g #2.*(g - np.min(g))/np.ptp(g)-1


def run_iir_task(impulse_responses_json, debug=False):
    impulseResponses = [loads(bytes(ir_json, 'latin1')) for ir_json in impulse_responses_json] if not debug else impulse_responses_json
    ir = np.mean(impulseResponses, axis=0)
    g = compute_filter_g(ir)
    return g.tolist()