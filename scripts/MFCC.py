import numpy as np
from scipy.fftpack import dct

def pre_emphasis(signal, alpha=0.97):
    return np.append(signal[0], signal[1:] - alpha * signal[:-1])

def framing_exact_fps(signal, sample_rate, fps=30):
    frame_len = int((1.0/fps) * sample_rate)       # 33.333 ms
    frame_step = frame_len                         # NO OVERLAP
    
    signal_length = len(signal)
    num_frames = signal_length // frame_step   # truncate extra tail

    frames = []
    for i in range(num_frames):
        start = i * frame_step
        end = start + frame_len
        frames.append(signal[start:end])

    return np.array(frames)

def hamming_window(frames):
    win = np.hamming(frames.shape[1])
    return frames * win

def power_spectrum(frames, NFFT=512):
    mag_frames = np.abs(np.fft.rfft(frames, NFFT))
    return (1.0 / NFFT) * (mag_frames ** 2)

def mel_filterbanks(sample_rate, nfilt=40, NFFT=512):
    low_mel = 0
    high_mel = 2595 * np.log10(1 + (sample_rate/2) / 700)

    mel_points = np.linspace(low_mel, high_mel, nfilt + 2)
    hz_pts = 700 * (10**(mel_points/2595) - 1)
    bins = np.floor((NFFT + 1) * hz_pts / sample_rate).astype(int)

    fbank = np.zeros((nfilt, int(NFFT/2 + 1)))
    for i in range(1, nfilt + 1):
        left = bins[i-1]
        center = bins[i]
        right = bins[i+1]

        for k in range(left, center):
            fbank[i-1][k] = (k - left) / (center - left)
        for k in range(center, right):
            fbank[i-1][k] = (right - k) / (right - center)
    return fbank

def mfcc_30fps(signal, sr, num_ceps=13, nfilt=40, NFFT=512):
    emphasized = pre_emphasis(signal)
    frames = framing_exact_fps(emphasized, sr, fps=30)
    frames = hamming_window(frames)

    power = power_spectrum(frames, NFFT)
    fbanks = mel_filterbanks(sr, nfilt, NFFT)
    energies = np.dot(power, fbanks.T)

    energies = np.where(energies == 0, np.finfo(float).eps, energies)
    log_energy = np.log(energies)

    ceps = dct(log_energy, type=2, axis=1, norm='ortho')[:, :num_ceps]
    return ceps
