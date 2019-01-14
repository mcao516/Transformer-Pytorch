# -*- coding: utf-8 -*-

"""Configuration class.

   Author: Meng Cao
"""

import os

from datetime import datetime
from .general_utils import get_logger


class Config(): 

    def __init__(self, operation=""):
        """Initialize hyperparameters and load vocabs.
        """
        self.dir_output = "results/{}/{:%Y%m%d_%H%M%S}/".format(operation, 
            datetime.now())
        self.dir_model  = self.dir_output + "model/"
        self.path_log   = self.dir_output + "log.txt"
        self.path_params  = self.dir_output + "params.txt"
        self.path_summary = self.dir_output + "summary"
        
        # directory for training outputs
        if not os.path.exists(self.dir_model):
            os.makedirs(self.dir_model)

        # create instance of logger
        self.logger = get_logger(self.path_log)

    # data paths
    sample_path = ""
    train_dataset_path = ""
    dev_dataset_path  = ""
    test_dataset_path = ""

    # model parameters
    # 

    # hyper parameters
    d_model = 512
    N = 6 # the total number of 
    head_num = 8
    d_ff = d_model * 4
    vocab_size = 11

    def write_params(self, file_path=None):
        """Save model parameters to a file.
        """
        raise NotImplementedError
