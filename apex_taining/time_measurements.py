import time
import torch

import time
import numpy as np
import torch
torch.backends.cudnn.benchmark=False
import argparse
import os
import sys

import sys
sys.path.append('/trinity/home/y.gusak/ReducedOrderNet/src')
from models.dcp.pruned_resnet import PrunedResNet

sys.path.append('/trinity/home/y.gusak/musco')
from flopco import FlopCo

from torchvision.models import resnet18


def measure_time(model, image, n_, repeats_=10):
    time_dict = {}
    for test_device in ['cpu', 'cuda']:
        model = model.to(test_device).eval()
        image = image.to(test_device)
        print(image.shape)
        est_time = []
        n = n_
        repeats = repeats_
        if test_device is 'cuda':
            a = torch.randn(3000, 3000).cuda() @ torch.randn(3000, 3000).cuda()
            n = n_*2
            repeats = repeats_*10

        for i in range(n):
            ex_time = 0
            for j in range(repeats):
                image = torch.randn(image.shape).to(test_device)
                start = time.time()
                model(image)
                ex_time += (time.time()-start)
                if test_device is 'cuda':
                    torch.cuda.synchronize()
            ex_time = ex_time/repeats
            est_time.append(ex_time)
            print('hop')
            
        time_dict[test_device] = {'mean': np.mean(est_time), 'std': np.std(est_time)}
        print('{}'.format(time_dict))
    return time_dict


if __name__=="__main__":
    
    mpath = "/gpfs/gpfs0/y.gusak/musco_models/resnet18_imagenet/grschedule1/tucker2_vbmf_wf0.8/iter_0-7/beforeft.pth"
    spath = "/gpfs/gpfs0/y.gusak/musco_models/resnet18_imagenet/grschedule1/tucker2_vbmf_wf0.8/iter_0-7/best.pth.tar"

    mym = torch.load(mpath)
    mym.load_state_dict(torch.load(spath, map_location = 'cpu')['state_dict'])
    
    
    print(mym)
    
    n = 5
    image = torch.randn(1, 3, 224, 224)
    
    mym_time = measure_time(mym, image, n)
    
    m = resnet18(pretrained = True)
    m_time = measure_time(m, image, n)
    
    mym_flopco = FlopCo(mym)
    m_flopco = FlopCo(m)
    
    
    for d in ['cpu', 'cuda']:
        print(d, m_time[d]['mean']/mym_time[d]['mean'])
        
    print('flops', m_flopco.total_flops/mym_flopco.total_flops)
    
    
    

    
    