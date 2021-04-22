# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 11:55:29 2021

@author: Maciej
"""

import numpy as np


def function4(x):
    if x<0:
        return np.sin(x)
    else:
        return np.sqrt(x)
