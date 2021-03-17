import torch
import torchvision.transforms as T
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.utils import save_image
import core as Co
import utils as U
from model.resnet import resnet50,lambdaresnet50
import os
import pickle as pkl
def operate(phase):
    if phase=='train':
        model.train()
        loader=trainloader
    else:
        model.eval()
        loader=valloader
    for i,(data,target)in enumerate(loader):
        data=data.to(device)
        target=target.to(device)
        output=model(data)
        loss=lossf(output,target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        acc=U.acc(output,target)
        print(f'{e}/{epoch}:{i}/{len(loader)}, loss:{loss:.2f},acc":{acc:.2f}')
        Co.addvalue(writer,'loss',loss.item(),e)
        Co.addvalue(writer,'acc',acc.item(),e)

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--batchsize',default=8,type=int)
    parser.add_argument('--model',default='resnet')
    parser.add_argument('--dataset',default='cifar100')
    parser.add_argument('--optimizer',default='adam')
    parser.add_argument('--epoch',default=100,type=int)
    parser.add_argument('--savefolder',default='tmp')
    parser.add_argument('--checkpoint',default=None)
    parser.add_argument('--size',default=64,type=int)
    parser.add_argument('--feature',default=128,type=int)
    args=parser.parse_args()
    epoch=args.epoch
    device='cuda' if torch.cuda.is_available() else 'cpu'
    savefolder='data/'+args.savefolder
    os.makedirs(savefolder,exist_ok=True)
    if args.checkpoint:
        chk=torch.load(args.checkpoint)
        loader=chk['loader']
        model=chk['model']
        e=chk['e']
        writer=chk['writer']
        args=chk['args']
    else:
        if args.dataset=='cifar100':
            trainloader=torch.utils.data.DataLoader(torchvision.datasets.CIFAR10('../data',True,T.Compose([T.Resize(args.size),T.ToTensor()]),download=True),batch_size=args.batchsize,num_workers=4,shuffle=True)
            valloader=torch.utils.data.DataLoader(torchvision.datasets.CIFAR10('../data',False,T.Compose([T.Resize(args.size),T.ToTensor()]),download=True),batch_size=args.batchsize,num_workers=4,shuffle=True)
            num_classes=101
        else:
            assert False,'cifar100 is allowed only.'
        if args.optimizer=='adam':
            _optimizer=torch.optim.Adam
        if args.model == 'resnet':
            model=resnet50(num_classes=num_classes)
        elif args.model=='lambdaresnet':
            model=lambdaresnet50(num_classes=num_classes)
        optimizer=_optimizer(model.parameters())
        writer={}
        e=0
    import json
    with open(f'{savefolder}/args.json','w') as f:
        json.dump(args.__dict__,f)
    lossf=nn.CrossEntropyLoss()
    if device=='cuda':
        model=model.to(device)
    for e in range(e,epoch):
        operate('train')
        operate('val')
        torch.save({
            'model':model.to('cpu'),
            'e':e,
            'writer':writer,
            'args':args,
            'loader':loader
        },savefolder+'/chk.pth')
        model.to(device)
        Co.savedic(writer,savefolder,"")