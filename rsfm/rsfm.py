# -*- coding: utf-8 -*-

from keras.layers import Input
from keras.layers.recurrent import GRU
from keras.models import Sequential
from keras.models import Model


class Rsmf(object):
    def __init__(self, x_train, training_params):
        self.x_train = x_train
        self.epoch = training_params["epoch"]
        self.batch_size = training_params["batch_size"]

    def train(self):
        pass
