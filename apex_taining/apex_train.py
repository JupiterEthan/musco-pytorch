#!/usr/bin/env python
# coding: utf-8


import os
import sys
# sys.path.append("../../src")

import argparse
gpus = "0,1,2,3"
num_gpus = len(gpus.split(','))

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"]=gpus

import time
import torch
from torch import nn
import numpy as np
from itertools import combinations
from collections import OrderedDict,defaultdict

from torchvision import datasets, transforms
from torchvision.models import resnet50

from apex import amp
# FOR DISTRIBUTED: (can also use torch.nn.parallel.DistributedDataParallel instead)
from apex.parallel import DistributedDataParallel


parser = argparse.ArgumentParser()
# FOR DISTRIBUTED:  Parse for the local_rank argument, which will be supplied
# automatically by torch.distributed.launch.
parser.add_argument("--local_rank", default=0, type=int)
args = parser.parse_args()

args.distributed = False
if 'WORLD_SIZE' in os.environ:
    args.distributed = int(os.environ['WORLD_SIZE']) > 1


if args.distributed:
    # FOR DISTRIBUTED:  Set the device according to local_rank.
    torch.cuda.set_device(args.local_rank)

    # FOR DISTRIBUTED:  Initialize the backend.  torch.distributed.launch will provide
    # environment variables, and requires that you use init_method=`env://`.
    torch.distributed.init_process_group(backend='nccl',
                                         init_method='env://')
    args.world_size = torch.distributed.get_world_size()

    
    
dataset = OrderedDict()


def accuracy(outputs, targets, topk=(1, )):
    """
        Computes the accuracy@k for the specified values of k
        src: https://catalyst-team.github.io/catalyst/_modules/catalyst/dl/metrics.html
    """
    max_k = max(topk)
    batch_size = targets.size(0)

    _, pred = outputs.topk(max_k, 1, True, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def fast_collate(batch):
    imgs = [img[0] for img in batch]
    targets = torch.tensor([target[1] for target in batch], dtype=torch.int64)
    w = imgs[0].size[0]
    h = imgs[0].size[1]
    tensor = torch.zeros( (len(imgs), 3, h, w), dtype=torch.uint8 )
    for i, img in enumerate(imgs):
        nump_array = np.asarray(img, dtype=np.uint8)
        if(nump_array.ndim < 3):
            nump_array = np.expand_dims(nump_array, axis=-1)
        nump_array = np.rollaxis(nump_array, 2)

        tensor[i] += torch.from_numpy(nump_array)
        
    return tensor, targets





data_root = "/gpfs/gpfs0/e.ponomarev"
data_name = 'imagenet'

args.batch_size = 256
args.num_workers = 64

num_epochs = 20
log_interval = 1


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

traindir = os.path.join(data_root, data_name, 'train')
valdir = os.path.join(data_root, data_name, 'val')
    
dataset['train'] = datasets.ImageFolder(
                traindir,
                transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    #transforms.ToTensor(),
                    #normalize,
                ]))

dataset['val']  = datasets.ImageFolder(valdir, 
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                #transforms.ToTensor(),
                #normalize,
            ]))


device = 'cuda'
model = resnet50(pretrained = True)
model = model.to(device)

args.lr = 0.0001
args.momentum = 0.9
args.weight_decay = 1e-4
args.lr = args.lr*float(args.batch_size*args.world_size)/256. 
optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum, weight_decay=args.weight_decay)
criterion = torch.nn.CrossEntropyLoss().to(device)
#optimizer = torch.optim.Adam(model.parameters(),weight_decay=1e-6,lr=lr)
#scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)


model, optimizer = amp.initialize(model, optimizer, opt_level="O2")
model = DistributedDataParallel(model)
#model = torch.nn.parallel.DistributedDataParallel(model,
#                                                   device_ids=[args.local_rank],
#                                                   output_device=args.local_rank)


logdir = './logdir/'
if not os.path.exists(logdir):
    os.mkdir(logdir)

# Partition dataset among workers using DistributedSampler
train_sampler = torch.utils.data.distributed.DistributedSampler(dataset['train'])
train_loader = torch.utils.data.DataLoader(dataset['train'], batch_size=args.batch_size, 
                                           sampler=train_sampler,num_workers=args.num_workers, collate_fn = fast_collate, pin_memory = True)

val_sampler = torch.utils.data.distributed.DistributedSampler(dataset['val'])
val_loader = torch.utils.data.DataLoader(dataset['val'], batch_size=args.batch_size, sampler=val_sampler,num_workers=args.num_workers, collate_fn = fast_collate, pin_memory = True)



if args.local_rank == 0:
    start_t = time.time()

im_mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).cuda().view(1,3,1,1)
im_std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).cuda().view(1,3,1,1)



for epoch in range(num_epochs):
    y_true = []
    y_pred = []
    for batch_idx, (data, target) in enumerate(train_loader):
        model.train()
        optimizer.zero_grad()
        data = data.float().to(device).sub_(im_mean).div_(im_std)
        #data = data.to(device)
        target = target.to(device)
        output = model(data)
        #pred = torch.argmax(output,1)
        loss = criterion(output, target)
        y_true.append(target)
        y_pred.append(output)
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0 and args.local_rank == 0:
            now_time = time.time()
            t = int(now_time-start_t)
            print("Time:\t%d min\t%d seconds" % ( t//60,t%60))
            print('Train Epoch: {} [{}/{}] batches\tLoss: {}'.format(
               
                epoch, (batch_idx * len(data)*num_gpus)//args.batch_size, num_gpus*len(train_sampler)//args.batch_size, loss.item()))
        
    
    #y_true = torch.cat(y_true)
    #y_pred = torch.cat(y_pred)
    #with torch.no_grad():
    #    accuracy_score = torch.stack(accuracy(y_pred,y_true,topk=(1,5))).cpu().numpy().astype('float32')
    #    accuracy_score = np.squeeze(accuracy_score)
    #print("device %d, epoch %d, train acc score %s" % (args.local_rank,epoch,accuracy_score))
    # validation
    #torch.cuda.empty_cache()
    with torch.no_grad():
        model.eval()
        y_true = []
        y_pred = []
        for batch_idx, (data, target) in enumerate(val_loader):

            data = data.float().to(device).sub_(im_mean).div_(im_std)
            #data = data.to(device)
            target = target.to(device)

            output = model(data)
            #pred = torch.argmax(output,1)
            
            y_true.append(target.cpu())
            y_pred.append(output.cpu())
            loss = criterion(output, target)
            if batch_idx % log_interval == 0 and args.local_rank == 0:
                now_time = time.time()
                t = int(now_time-start_t)
                print("Time:\t%d min\t%d seconds" % ( t//60,t%60))
                print('Val Epoch: {} [{}/{}] batches\tLoss: {}'.format(
                    epoch, (batch_idx * len(data)*num_gpus)//args.batch_size, num_gpus*len(val_sampler)//args.batch_size, loss.item()))
        
        y_true = torch.cat(y_true)
        y_pred = torch.cat(y_pred)
        val_acc = torch.stack(accuracy(y_pred,y_true,topk=(1,5))).numpy().astype('float32')
        val_acc = np.squeeze(val_acc)
        print("Device %d, Epoch %d, val acc score %s" % (args.local_rank,epoch,val_acc))
        if args.local_rank == 0:
            torch.save(model.state_dict(), '%s/%d-apex_model_%s_%s.pt' % (logdir,epoch,val_acc[0],val_acc[1])
                      )
