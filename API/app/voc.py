# -*- coding: utf-8 -*-
"""
Created on Sat Nov 27 09:14:26 2021

@author: mitran
"""

import torch
from torch.nn import Module,Conv1d,BatchNorm1d,ReLU,Sigmoid,LSTMCell,Linear,ModuleList,Upsample,Embedding,Sequential,AvgPool1d,Dropout,LeakyReLU,Tanh,BCELoss,LSTM,GRU,ConvTranspose1d,MSELoss
from torch.nn.utils import weight_norm

from torch.nn import functional as F
from torch.optim import Adam


import zipfile
import librosa
from scipy.io.wavfile import read


import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import re
import IPython.display as ipd
from torch.autograd import Variable
import random
from torch.utils.data import DataLoader






    
def Feature_Extractor(config):
  # extract some features from spectrogram so generator can work on it
  return Sequential(
      weight_norm(Conv1d(config.spectrogram_dimension,config.feature_dim//4,7,padding=3)),
      LeakyReLU(0.2),
      weight_norm(Conv1d(config.feature_dim//4,config.feature_dim,7,padding=3)),
      LeakyReLU(0.2),
     
  )
class Resnet_block(Module):
  def __init__(self,in_dim,out_dim=None,dilation=1):

    super().__init__()

    if(out_dim==None):
      out_dim=in_dim


    # padding must be equal to dilation wihen kernel is 3

    self.conv1=weight_norm(Conv1d(in_dim,out_dim,5,dilation=dilation,padding=dilation*2))
    self.act1= LeakyReLU(0.2,inplace=True)

    self.conv2=weight_norm(Conv1d(out_dim,out_dim,1))

    self.skip=weight_norm(Conv1d(in_dim,out_dim,1))

    self.act2= LeakyReLU(0.2,inplace=True)

  def forward(self,X):

    y=self.act1(self.conv1(X))

    y=self.conv2(y)

    y1=self.skip(X)
    
    op=self.act2(y+y1)

    return op

class GeneratorBlock(Module):
  def __init__(self,in_dim,out_dim,scale):
    # upsampling and conv block  conv will be followed by tranposed conv bcz to increase the receptive field after upsampling

    # using transpose2d for upsampling

    # 2*stride ==kernle will give proper upsample where stride is the scale of upsampling and padding =scale //2 makes it exact

    super().__init__()

    self.upsample=weight_norm(ConvTranspose1d(in_dim,out_dim,scale*2,scale,padding=scale//2))
    self.act=LeakyReLU(0.2,inplace=True)

    # convolution to increase receptive filed 
    # upsampled and its audio need high receptive 
    # dilation increase the receptive field exponentially

    # 1,3,9 dilation are choosen carefully for good output ->paper

    dia=[1,5,25]

    self.increase_receptivefield = Sequential(
        
                Resnet_block(out_dim,dilation=dia[0]),
                Resnet_block(out_dim,dilation=dia[1]),
                Resnet_block(out_dim,dilation=dia[2]),

        
    )


  def forward(self,X):


      y=self.upsample(X)
      y=self.act(y)

      # after upsampling receptive field would be decreased so individual feature ill have less information about the input

      y=self.increase_receptivefield(y)

      return y




    



    





class GenerativeNetwork(Module):
  def __init__(self,config):

    super().__init__()
    self.feature_extract=Feature_Extractor(config)

    scale=config.Generator_Scale
    dimensions=config.Generator_dimensions
    temp=ModuleList()

    for i in range(len(config.Generator_dimensions)):
      
      temp.append(GeneratorBlock(config.feature_dim if i==0 else dimensions[i-1]  , dimensions[i] ,scale[i]))
    self.upsample_Modules=temp

    self.final_conv=Conv1d(dimensions[-1],config.audio_dim,1)
                               

  def forward(self,X):


    y=self.feature_extract(X)
    for layer in self.upsample_Modules:
      
      y=layer(y)
    y=self.final_conv(y)
    y=torch.tanh(y)
    

    return y




    