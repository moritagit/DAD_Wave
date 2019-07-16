# -*- coding: utf-8 -*-


from pathlib import Path

import numpy as np
import pandas as pd

import librosa
import torch

import preprocess


class ESC50Dataset(torch.utils.data.Dataset):
    def __init__(self, metadata_path, audio_dir, spectrogram_dir, transform=None,):
        self.transform = transform

        self.metadata = pd.read_csv(str(metadata_path))
        self.audio_dir = Path(audio_dir)
        self.spectrogram_dir = Path(spectrogram_dir)

        self.label_data = None
        self.labels = set()
        self.label2indices = {}

        self._build()

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        return self.load_audio(index)

    def load_audio(self, index):
        fname = self.metadata.at[index, 'filename']
        fpath = self.audio_dir/fname
        x, fs = librosa.load(str(fpath))
        return x, fs

    def load_spectrogram(self, index):
        fname = self.metadata.at[index, 'fname']+'.npy'
        fpath = self.spectrogram_dir/fname
        x = np.load(str(fpath))
        return x

    def load_label(self, index):
        label = self.metadata.at[index, 'target']
        return label

    def _build_label_data(self):
        label_data = self.metadata.loc[:, ['target', 'category']].drop_duplicates()
        label_data = label_data.sort_values(by=['target'], ascending=True)
        label_data = label_data.reset_index(drop=True)

        label_data['number'] = 0
        label2indices = {}
        for i, target in enumerate(label_data['target']):
            data = self.metadata.query('target == @target')
            label_data.loc[i, 'number'] = len(data)
            label2indices[target] = data.index.values.tolist()

        self.labels = set(label2indices.keys())
        self.label_data = label_data
        self.label2indices = label2indices
        return label_data, label2indices

    def _build_spectrogram(self):
        self.metadata['fname'] = ''
        for index in range(len(self.metadata)):
            audio_file_path = self.audio_dir/self.metadata.at[index, 'filename']
            fname = audio_file_path.stem
            self.metadata.at[index, 'fname'] = fname

            fname += '.npy'
            spec_file_path = self.spectrogram_dir/fname
            if not spec_file_path.exists():
                x, fs = self.load_audio(index)
                spec = preprocess.spectrogram(x, n_fft=512, hop_length=256)
                np.save(str(spec_file_path), spec)
        return

    def _build(self):
        self._build_label_data()
        self._build_spectrogram()


class ESC50DatasetTriplet(ESC50Dataset):
    def __init__(self, metadata_path, audio_dir, spectrogram_dir, transform=None,):
        super(ESC50DatasetTriplet, self).__init__(metadata_path, audio_dir, spectrogram_dir, transform)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        spec_anc, label_anc = self.load_spectrogram(index), self.load_label(index)

        # positive sampling
        indices_pos = self.label2indices[label_anc]
        index_pos = index
        if len(indices_pos) > 1:
            while index_pos == index:
                index_pos = np.random.choice(indices_pos)
        spec_pos = self.load_spectrogram(index_pos)

        # negative sampling
        labels_neg = list(self.labels - set([label_anc]))
        label_neg = np.random.choice(labels_neg)
        index_neg = np.random.choice(self.label2indices[label_neg])
        spec_neg = self.load_spectrogram(index_neg)

        if self.transform is not None:
            spec_anc = self.transform(spec_anc)
            spec_pos = self.transform(spec_pos)
            spec_neg = self.transform(spec_neg)

        return (spec_anc, spec_pos, spec_neg), []
