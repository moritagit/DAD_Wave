# -*- coding: utf-8 -*-


import matplotlib.pyplot as plt

import librosa.display


def visualize_spectrogram(sp, fs, hop_length=128, mel=False, path=None):
    # display wave in spectrogram
    if mel:
        y_axis = 'mel'
        title = 'Mel Spectrogram'
        hop_length = 128
    else:
        y_axis = 'log'
        title = 'Spectrogram'
        hop_length = hop_length

    librosa.display.specshow(sp, sr=fs, x_axis='time', y_axis=y_axis, hop_length=hop_length)
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    if path:
        plt.savefig(str(path))
    plt.show()
