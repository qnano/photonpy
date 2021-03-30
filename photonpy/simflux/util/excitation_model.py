# -*- coding: utf-8 -*-
import numpy as np

def compute(x,y,mod):
    return mod[...,4]*(1+mod[...,2]*np.sin(mod[...,0]*x + mod[...,1]*y - mod[...,3]))

