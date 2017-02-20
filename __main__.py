# -*- coding: utf-8 -*-

import argparse
import yaml
from file_io.wave_io import WaveInput
from file_io.wave_io import WaveOutput


class Main(object):
    def __init__(self, args):
        settings_filename = "settings/" + args.settings
        with open(settings_filename, 'r') as f:
            self.settings = yaml.load(f)

    def __call__(self):
        winput = WaveInput(self.settings, num_file=10)
        woutput = WaveOutput(self.settings)
        woutput.write(winput.input_waves)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s", "--settings",
        help="the path of a settings file",
        default="settings.yml"
    )
    args = parser.parse_args()

    Main(args)()
