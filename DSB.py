import numpy as np
from scipy.signal import butter, lfilter, hilbert


def modulation_dsbsc(original_audio, carrier, sr):
    modulated = original_audio * carrier
    
    N = len(modulated)
    
    fft_modulated = np.fft.fft(modulated)
    
    freq = np.fft.fftfreq(N, 1/sr)
    magnitude = np.abs(fft_modulated) / N
    pos_idx = np.where(freq >= 0)
    
    return modulated, freq[pos_idx], magnitude[pos_idx]
    # modulated is time domain modulated signal
    #freq[pos_idx] is x axis for frequency domain
    #mag is y axix for frequency domain


def demodulation_dsbsc(signal, carrier, sr, BW, gain=1.0, SNR_dB=0):

    # -----------------------------------
    signal_power = np.mean(signal**2)
    noise_power = signal_power / (10**(SNR_dB / 10))
    noise = np.sqrt(noise_power) * np.random.randn(len(signal))
    signal = signal + noise
    # -----------------------------------

    mixed = signal * carrier

    nyq = 0.5 * sr
    normal_cutoff = BW / nyq
    b, a = butter(N=5, Wn=normal_cutoff, btype='low', analog=False)
    demodulated = gain * lfilter(b, a, mixed)

    N = len(demodulated)
    fft_demod = np.fft.fft(demodulated)
    freq = np.fft.fftfreq(N, 1/sr)
    magnitude = np.abs(fft_demod) / N

    pos_idx = np.where(freq >= 0)
    return demodulated, freq[pos_idx], magnitude[pos_idx]


def modulation_dsbtc(original_audio, carrier, sr,DC_shitf) :
    modulated = (original_audio + DC_shitf) * carrier
    
    N = len(modulated)
    
    fft_modulated = np.fft.fft(modulated)
    
    freq = np.fft.fftfreq(N, 1/sr)
    magnitude = np.abs(fft_modulated) / N
    pos_idx = np.where(freq >= 0)
    
    return modulated, freq[pos_idx], magnitude[pos_idx]  


def demodulation_dsbtc(signal, sr, BW, gain=1.0, SNR_dB=0, DC_cutoff=20):
    signal_power = np.mean(signal**2)
    noise_power = signal_power / (10**(SNR_dB / 10))
    noise = np.sqrt(noise_power) * np.random.randn(len(signal))
    signal = signal + noise

    analytic = hilbert(signal)
    envelope = np.abs(analytic)

    nyq = 0.5 * sr
    normal_cutoff = BW / nyq
    b, a = butter(N=5, Wn=normal_cutoff, btype='low', analog=False)
    demodulated = gain * lfilter(b, a, envelope)

    normal_cutoff_dc = DC_cutoff / nyq
    b_dc, a_dc = butter(N=2, Wn=normal_cutoff_dc, btype='high', analog=False)
    demodulated = lfilter(b_dc, a_dc, demodulated)

    N = len(demodulated)
    fft_demod = np.fft.fft(demodulated)
    freq = np.fft.fftfreq(N, 1/sr)
    magnitude = np.abs(fft_demod) / N
    pos_idx = np.where(freq >= 0)
    return demodulated, freq[pos_idx], magnitude[pos_idx]

