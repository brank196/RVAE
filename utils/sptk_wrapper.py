# -*- coding: utf-8 -*-

import subprocess
import os
import numpy as np
from file_io.wave_io import Wave
from file_io.wave_io import WaveInput
from file_io.wave_io import WaveOutput


class SptkWrapper(object):
    @classmethod
    def execute(cls, cmd, split=True):
        if split:
            cmd = cmd.split()
        subprocess.check_call(cmd, shell=True)

    @classmethod
    def remove_all(cls):
        filenames = os.listdir("tmp")
        for filename in filenames:
            os.remove("tmp/" + filename)

    @classmethod
    def remove_one(cls, filename):
        try:
            os.remove("tmp/" + filename)
        finally:
            pass

    @classmethod
    def wav2raw(cls, wave):
        assert isinstance(wave, Wave)

        WaveOutput.write(wave, "tmp")

        infile_path = "tmp/" + wave.filename
        if wave.filename.endswith(".wav"):
            raw_path = "tmp/" + wave.filename.replace(".wav", ".raw")
        else:
            raw_path = "tmp/" + wave.filename + ".raw"
        cmd = "wav2raw -d tmp " + infile_path
        cls.execute(cmd)

        return raw_path

    @classmethod
    def pitch(cls, raw_path, a=1, s=16, p=80, t=0.3, L=60, H=240, O=0):
        assert isinstance(raw_path, str)

        pitch_path = "tmp/" + os.path.basename(raw_path).replace(".raw", ".pitch")
        dmp_path = "tmp/" + os.path.basename(raw_path).replace(".raw", ".txt")
        option = (
            "-a {0} -s {1} -p {2} -t{0} {3} -L {4} -H {5} -o {6}"
        ).format(a, s, p, t, L, H, O)
        cmd = (
            "x2x +sf {0} | pitch {1} > {2}"
        ).format(raw_path, option, pitch_path)
        cls.execute(cmd)

        cmd = "dmp +f {0} > {1}".format(pitch_path, dmp_path)
        cls.execute(cmd)

        with open(dmp_path, 'r') as fdmp:
            pitch = np.array([float(val.rstrip()) for val in fdmp], dtype=np.float32)

        return pitch, pitch_path

    @classmethod
    def excite(cls, pitch_path, p=100, i=1, n=False, s=1):
        option = "-p {0} -i {1} {2} -s {3}".format(
            p, i, "-n" if n else "", s
        )
        source_path = "tmp/" + os.path.basename(pitch_path).replace(".pitch", ".source.wav")
        cmd = "excite {0} {1} > {2}".format(option, pitch_path, source_path)
        cls.execute(cmd)

        return WaveInput.read(source_path)

    @classmethod
    def easy_pitch(cls, wave):
        raw_path = cls.wav2raw(wave)
        return cls.pitch(raw_path)

    @classmethod
    def easy_excite(cls, wave):
        _, pitch_path = cls.easy_pitch(wave)
        return cls.excite(pitch_path)

