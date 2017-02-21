# -*- coding: utf-8 -*-

import os
import random
import numpy as np
from scipy.io import wavfile
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
        self.settings = settings
        self.input_dirname = self.settings["INPUT_DIRNAME"]
        self.num_file = num_file
        self.input_waves = []
        if self.num_file is not None:
            filenames = os.listdir(self.input_dirname)
            len_filenames = len(filenames)
            if self.num_file == 'all':
                self.num_file = len_filenames
            if self.num_file > len_filenames:
                raise ValueError(
                    "too many num_file ({0} > {1})".format(self.num_file, len_filenames)
                )

            random.shuffle(filenames)
            self.input_waves = self.read_waves(filenames[:self.num_file])

    def read_waves(self, filepathes):
        if not isinstance(filepathes, list):
            filepathes = [filepathes]
        input_waves = [self.read(filepath) for filepath in filepathes]
        return input_waves

    @staticmethod
    def read(filepath):
        rate, data = wavfile.read(filepath)
        return Wave(filename=os.path.basename(filepath), rate=rate, data=data)

    @staticmethod
    def make_batch(input_waves, batch_size):
        len_input_waves = len(input_waves)

        def batch_generator(_batch_size):
            batch = []
            for input_file in input_waves:
                batch.append(input_file)
                if len(batch) == _batch_size:
                    yield batch
                batch = []
            for _ in range(_batch_size - len(batch)):
                batch.append(random.choice(input_waves))
            yield batch
        # keras先生がバッチ処理やってくれるっぽいけど...
        if batch_size > len_input_waves:
            raise ValueError(
                "batch_size is larger than #data ({0} > {1})".format(batch_size, len_input_waves)
            )
        batches = [x for x in batch_generator(batch_size)]
        return batches


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

    def write_waves(self, output_waves):
        if not isinstance(output_waves, list):
            output_waves = [output_waves]
        for output_wave in output_waves:
            self.write(output_wave, self.output_dirname)

    @staticmethod
    def write(wave, dirname):
        assert isinstance(wave, Wave)
        filepath = dirname + "/" + wave.filename
        wavfile.write(filepath, wave.rate, wave.data)
