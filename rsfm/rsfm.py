# -*- coding: utf-8 -*-

import logging
from keras.layers import Input
from keras.layers.recurrent import GRU
from keras.models import Sequential
from keras.models import Model
from utils.sptk_wrapper import SptkWrapper


class Rsfm(object):
    def __init__(self, x_train, training_params):
        self.x_train = x_train
        self.sources = [SptkWrapper.easy_excite(wave) for wave in self.x_train]
        self.epoch = training_params["epoch"]
        self.batch_size = training_params["batch_size"]
        logging.getLogger(__name__)
        fmt = "%(asctime)s - %(name)s - %(message)s"
        logging.basicConfig(format=fmt)
        self.logger = logging

    def do_epoch(self):
        return 0

    def train(self):
        for epoch in range(self.epoch):
            self.logger.info("epoch {0} / {1} start".format(epoch, self.epoch))
            avg_loss = self.do_epoch()
            self.logger.info("average loss={0:.4e}".format(avg_loss))
