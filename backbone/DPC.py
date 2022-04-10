import sys
import time
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
sys.path.append('..')
from backbone.select_backbone import select_resnet
from backbone.convrnn import ConvGRU

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class DPC_RNN(nn.Module):
    '''DPC with RNN'''
    def __init__(self, sample_size, num_seq=6, seq_len=5, pred_step=1, network='resnet18'):
        # sample_size is size of input images, i.e. (128,128)
        super(DPC_RNN, self).__init__()
        torch.cuda.manual_seed(233)
        print('Using DPC-RNN model')
        self.sample_size = sample_size
        self.num_seq = num_seq
        self.seq_len = seq_len
        self.pred_step = 0
        self.last_duration = int(math.ceil(seq_len / 4))
        self.last_size = int(math.ceil(sample_size / 16))
        print('final feature map has size %dx%d' % (self.last_size, self.last_size))

        self.backbone, self.param = select_resnet(network, track_running_stats=False)
        self.param['num_layers'] = 1 # param for GRU
        self.param['hidden_size'] = self.param['feature_size'] # param for GRU

        self.agg = ConvGRU(input_size=self.param['feature_size'],
                               hidden_size=self.param['hidden_size'],
                               kernel_size=1,
                               num_layers=self.param['num_layers'])
        self.network_pred = nn.Sequential(
                                nn.Conv2d(self.param['feature_size'], self.param['feature_size'], kernel_size=1, padding=0),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(self.param['feature_size'], self.param['feature_size'], kernel_size=1, padding=0)
                                )
        self.mask = None
        self.relu = nn.ReLU(inplace=False)
        self._initialize_weights(self.agg)
        self._initialize_weights(self.network_pred)
        self.interaction_bank=nn.Parameter(torch.randn(self.param['feature_size'],self.last_size,self.last_size))
        print(f'interaction bank has size: {self.interaction_bank.shape}')

    def forward(self, block):
        # block: [B, N, C, SL, W, H]
        ### extract feature ###
        (B, N, C, SL, H, W) = block.shape
        block = block.view(B*N, C, SL, H, W)
        feature = self.backbone(block)
        del block
        feature = F.avg_pool3d(feature, (self.last_duration, 1, 1), stride=(1, 1, 1))

        feature = self.relu(feature) # [0, +inf)
        feature = feature.view(B, N, self.param['feature_size'], self.last_size, self.last_size) # [B,N,D,6,6], [0, +inf)

        ### aggregate, predict future ###
        _, hidden = self.agg(feature[:, 0:N-self.pred_step, :].contiguous())
        context = hidden[:,-1,:] # after tanh, (-1,1). get the hidden state of last layer, last time step
        pred = self.network_pred(context)
        # pred_pooled=F.adaptive_avg_pool2d(pred,output_size=(1,1)).squeeze(-1).squeeze(-1)
        # del pred
        del hidden
        del context

        return pred

    def _initialize_weights(self, module):
        for name, param in module.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.orthogonal_(param, 1)
        # other resnet weights have been initialized in resnet itself

    def reset_mask(self):
        self.mask = None


def load_backbone_state(model):
    state_dict_DataParallel=torch.load(os.path.join(args.exp_path,'EGO4D/SSL/ckpts/model_epoch_100.pth'),map_location='cpu')['model_state_dict']
    from collections import OrderedDict
    state_dict_normal=OrderedDict()
    for k,v in state_dict_DataParallel.items():
        key=k[7:]
        state_dict_normal[key]=v
    model.load_state_dict(state_dict_normal)
    return model

if __name__=='__main__':
    import os
    from opt import *
    device=torch.device('cpu')
    # model = DPC_RNN_Extractor(sample_size=128,
    #                 num_seq=6,
    #                 seq_len=5,
    #                 network='resnet18',
    #                 pred_step=1).to(device)
    # # block: [B, N, C, SL, W, H]
    # input=torch.randn((4,6,3,5,128,128))
    # context,_=model(input)
    # print('context shape:')
    # print(context.shape)

    model = DPC_RNN(sample_size=128,
                    num_seq=6,
                    seq_len=5,
                    network='resnet18',
                    pred_step=1).to(device)
    # block: [B, N, C, SL, W, H]
    model=load_backbone_state(model)
    input=torch.randn((4,5,3,5,128,128))
    pred=model(input)
    print('context shape:')
    print(pred.shape)
