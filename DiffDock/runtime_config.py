# -*- coding: utf-8 -*-
import os

class RuntimeConfig:
    DIFFDOCK_PATH = os.path.dirname(os.path.abspath(__file__))
    DEVICE = 'cuda:0'
    MODEL_PATH = os.path.join(DIFFDOCK_PATH, 'trained_weights')
    BATCH_SIZE = 32
    NUM_DIFFUSION_STEPS = 20