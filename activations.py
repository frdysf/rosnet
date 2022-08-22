#!/usr/bin/env python
# coding: utf-8

import numpy as np

def null(x):
    return x

def relu(x):
    return np.maximum(x, 0)
	
def sigmoid(x):  # NB: sometimes encounters overflow, even with dtype upgraded to np.longdouble
    return 1 / (1 + np.exp(-x))