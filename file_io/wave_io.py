# -*- coding: utf-8 -*-

import os
import random
import numpy as np
from scipy.io.wavfile import read
from scipy.io.wavfile import write
from scipy import signal


class Wave(object):
    __repr__ = __str__ = lambda self: "wave file of \"{0}\"".format(self._filename)

    def __init__(self, **kwargs):
        self._filename = None
        self._rate = None
        self._data = None
        if "filename" in kwargs:
            self.filename = kwargs["filename"]
        if "rate" in kwargs:
            self.rate = kwargs["rate"]
        if "data" in kwargs:
            self.data = kwargs["data"]

    @property
    def filename(self):
        return self._filename

    @filename.setter
    def filename(self, filename):
        self._filename = filename

    @property
    def rate(self):
        return self._rate

    @rate.setter
    def rate(self, rate):
        self._rate = rate

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        self._data = data


class WaveInput(object):
    def __init__(self, settings, num_file='all'):
        def batch_generator(input_waves, _batch_size):
            batch = []
            for input_file in input_waves:
                batch.append(input_file)
                if len(batch) == _batch_size:
                    yield batch
                batch = []
            for _ in range(_batch_size - len(batch)):
                batch.append(random.choice(input_waves))
            yield batch

        self.settings = settings
        self.input_dirname = self.settings["INPUT_DIRNAME"]
        filenames = os.listdir(self.input_dirname)
        len_filenames = len(filenames)
        if num_file == 'all':
            num_file = len_filenames
        if num_file > len_filenames:
            raise ValueError(
                "too many num_file ({0} > {1})".format(num_file, len_filenames)
            )

        sampling_params = self.settings["SAMPLING_PARAMETER"]
        window_msec = sampling_params["window_msec"]
        phrase_msec = sampling_params["phrase_msec"]
        if phrase_msec < window_msec / 2:
            raise ValueError("phrase_msec must be larger than half of window_msec")

        self.input_waves = []
        random.shuffle(filenames)
        for filename in filenames[:num_file]:
            filepath = self.input_dirname + '/' + filename
            rate, data = read(filepath)
            window_size = int((window_msec / 1000.) * rate)
            phrase_size = int((phrase_msec / 1000.) * rate)
            waves = self.__cut_waves(data, window_size, phrase_size)
            self.input_waves.append(
                Wave(filename=filename, rate=rate, data=waves)
            )
        """
        # keras先生がバッチ処理やってくれるっぽい
        batch_size = self.settings["TRAINING_PARAMETER"]["batch_size"]
        if batch_size > len_filenames:
            raise ValueError(
                "batch_size is larger than #data ({0} > {1})".format(batch_size, len_filenames)
            )
        self.batches = [
            x for x in batch_generator(self.input_waves, batch_size)
        ]
        """

    @staticmethod
    def __cut_waves(data, window_size, phrase_size):  # ハニング窓をかけて短時間ごとに切り分ける
        cuts = []
        size = data.shape[0]
        pad_size = phrase_size - (size - window_size) % phrase_size
        if pad_size < phrase_size:
            data = np.r_[data, np.zeros(pad_size)]
        hann = signal.hann(window_size)
        n0 = 0
        while n0 <= size - window_size:
            cuts.append(hann * data[n0:n0+window_size])
            n0 += phrase_size
        return cuts


class WaveOutput(object):
    def __init__(self, settings):
        self.settings = settings
        self.output_dirname = self.settings["OUTPUT_DIRNAME"]
        if not os.path.exists(self.output_dirname):
            os.makedirs(self.output_dirname)
        else:
            filenames = os.listdir(self.output_dirname)
            for filename in filenames:
                os.remove(self.output_dirname + "/" + filename)

    def write(self, output_waves):
        for output_wave in output_waves:
            filepath = self.output_dirname + "/" + output_wave.filename
            concat = self.__concat_waves(output_wave)
            write(filepath, output_wave.rate, concat)

    def __concat_waves(self, wave):  # 逆ハニング窓をかけて結合
        def mean_overlap(_data0, _data1):
            pad = np.zeros(phrase_size)
            _data0 = _data0 / hann
            _data1 = _data1 / hann
            _data0 = np.r_[_data0, pad]
            _data1 = np.r_[pad, _data1]
            return ((_data0 + _data1) / 2.)[:window_size]
        data = wave.data
        rate = wave.rate
        window_size = data[0].shape[0]
        concat = None
        sampling_params = self.settings["SAMPLING_PARAMETER"]
        phrase_msec = sampling_params["phrase_msec"]
        phrase_size = int((phrase_msec / 1000.) * rate)
        hann = signal.hann(window_size)        
        for data0, data1 in zip(data, data[1:]):
            wave = mean_overlap(data0, data1)
            if concat is None:
                concat = wave
            else:
                concat = np.r_[concat, wave]
        concat = np.r_[concat, data[-1][phrase_size:]]
        return concat
