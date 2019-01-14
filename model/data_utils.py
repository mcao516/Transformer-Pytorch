# -*- coding: utf-8 -*-

"""This python file contains mothods for data preparation.
   
   Author: Meng Cao
"""

import os
import random
import numpy as np


def minibatch(data_list, batch_size):
    """Generates data batch.
    """
    data_batch = []
    for data in data_list:
        data_batch += [data]
        # yield a batch of data
        if len(data_batch) == batch_size:
            yield data_batch
            data_batch = []
            
    if len(data_batch) > 0:
        yield data_batch
