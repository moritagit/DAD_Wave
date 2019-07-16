# -*- coding: utf-8 -*-


import numpy as np

import librosa


def spectrogram(x, n_fft=512, hop_length=256):
    # change wave data to stft
    stft = librosa.stft(x, n_fft=n_fft, hop_length=hop_length)
    sp = librosa.amplitude_to_db(np.abs(stft))
    return sp


def mel_spectrogram(x, n_fft=1024, hop_length=128):
    stft = np.abs(librosa.stft(x, n_fft=n_fft, hop_length=hop_length))**2
    log_stft = librosa.power_to_db(stft)
    melsp = librosa.feature.melspectrogram(S=log_stft,n_mels=128)
    return melsp


def add_white_noise(x, rate=0.002):
    return x + rate*np.random.randn(len(x))


def shift_sound(x, rate=2):
    return np.roll(x, int(len(x)//rate))


def stretch_sound(x, rate=1.1):
    input_length = len(x)
    x = librosa.effects.time_stretch(x, rate)
    if len(x) > input_length:
        return x[:input_length]
    else:
        return np.pad(x, (0, max(0, input_length - len(x))), 'constant')

