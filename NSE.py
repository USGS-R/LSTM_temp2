import scipy.io as sio
import os
import scipy.io as sio
import torch
import matplotlib.pyplot as plt

import numpy as np
import random
import numpy as np

def nanmean(v, mask, *args, inplace=False, **kwargs):
    if not inplace:
        v = v.clone()
    is_nan = torch.isnan(v)
    v[is_nan] = 0
    #m = v.sum(*args, **kwargs) / (~is_nan).float().sum(*args, **kwargs)
    m = v.sum(*args, **kwargs) / (mask).float().sum(*args, **kwargs)
    m[torch.isnan(m)] = 0
    m[~torch.isfinite(m)] = 0
    # if torch.isnan(m).sum() == len(m):
    #     m[torch.isnan(m)] = 0
    # else:
    #     m = m[~torch.isnan(m)]
    #     m[torch.isnan(m)] = 0
    return m

class NSELosstest(torch.nn.Module):
    # Same as Fredrick 2019
    def __init__(self):
        super(NSELosstest, self).__init__()

    def forward(self, output, target):
        Ngage = target.shape[1]
        losssum = 0
        nsample = 0
        target2 = target.clone()
        mask = target==target
        targetMean=nanmean(target2,axis=0).T.repeat([target.shape[0],1]).view(target.shape)

        target2[~mask] = 0
        output2 = output.clone(); output2[~mask]=0
        targetMean[~mask] = 0

        num = torch.sum((output2 - target2)**2,axis=0)
        denom = torch.sum((target2 - targetMean)**2,axis=0)

        maskSum = (denom==0)
        num[maskSum]=1
        denom[maskSum]=1
        NSE = 1-num/denom
        loss = torch.mean(NSE[~maskSum])
        return loss


class RMSEbasinLosstest(torch.nn.Module):
    # Same as Fredrick 2019
    def __init__(self):
        super(RMSEbasinLosstest, self).__init__()

    def forward(self, output, target):
        Ngage = target.shape[1]
        losssum = 0
        nsample = 0
        target2 = target.clone()
        output2 = output.clone()
       # A = (target2 - output2)**2
        mask = target2 == target2
        target2 = torch.where(target2 == target2, target2, torch.zeros(1).cuda())
        output2 = torch.where(target2 == target2, output2, torch.zeros(1).cuda())
        A = (target2 - output2) ** 2
        basinMean=(torch.sqrt(nanmean(A, mask, axis=0))).T.repeat([target.shape[0],1]).view(target.shape)
        basinMean[~mask] = 0
        loss = basinMean.mean()
        #targetMean = targetMean1.T.repeat([A.shape[0], 1]).view(A.shape)

        #targetMean = targetMean.T.repeat([target.shape[0], 1]).view(target.shape)
        #loss = (torch.sqrt(targetMean)).mean()
        #loss = ((targetMean)).mean()
        # A[~mask] = 0
        # mask1 = targetMean == targetMean
        # targetMean[~mask1] = 0
        # loss = (torch.sqrt(targetMean)).mean()
        # target2[~mask] = 0
        # #output2 = output.clone();
        # output2[~mask]=0
        # targetMean[~mask] = 0
        #
        # num = torch.sum((output2 - target2)**2,axis=0)
        # denom = torch.sum((target2 - targetMean)**2,axis=0)
        #
        # maskSum = (denom==0)
        # num[maskSum]=1
        # denom[maskSum]=1
        # NSE = 1-num/denom
        # loss = torch.mean(NSE[~maskSum])
        return loss


#loss = NSELosstest()
#l = loss(output,target)
