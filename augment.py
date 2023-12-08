##################################################################
####### **FROM https://github.com/zcaceres/spec_augment** ########
##################################################################


import torch
import random

import numpy as np
def freq_mask(spec, F=30, num_masks=1):
    cloned = spec.clone()
    num_mel_channels = cloned.shape[1]
    N = spec.shape[0]
    idx = torch.arange(N)
    
    for i in range(0, num_masks):        
        f = np.random.randint(0, F, N)
        f_0 = np.random.randint(0, num_mel_channels - f, N)
        lower = f_0
        upper = f_0 + f
        for j in range(N):
            cloned[j, lower[j]:upper[j],:] = cloned[j].mean()    
    return cloned

def time_mask(spec, T=40, num_masks=1):
    cloned = spec.clone()
    len_spectro = cloned.shape[2]
    N = spec.shape[0]
    
    for i in range(0, num_masks):
        t = np.random.randint(0, T, N)
        t_0 = np.random.randint(0, len_spectro - t, N)
        lower = t_0
        upper = t_0 + t

        for j in range(N):
            cloned[j, :, lower[j]:upper[j]] = cloned[j].mean()


    return cloned
