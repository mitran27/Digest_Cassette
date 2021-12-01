import zipfile
import librosa
from scipy.io.wavfile import read

import torch
from torch.nn import Module,Conv1d,BatchNorm1d,ReLU,Sigmoid,LSTMCell,Linear,ModuleList,Embedding,Sequential,Dropout,ReLU,Tanh,MSELoss,LSTM,GRU,AdaptiveAvgPool1d,CrossEntropyLoss
from torch.nn import functional as F
from torch.optim import Adam

import numpy as np
import pandas as pd
from tqdm import tqdm
import os
from num2words import num2words
import re
import IPython.display as ipd
from torch.autograd import Variable

from voc import GenerativeNetwork
path='seungwonpark/melgan'

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

def clean(text):
  x= re.sub(r"\([-^()]*\)", "",text)
  x=re.sub("""[^A-Za-z0-9]+""",' ',x)
  return x.lower()

class char_tokenizer():
  def __init__(self,space=True):
    self.tokendict={chr(ord('a')+i):i+1 for i in range(26)}
    if(space):self.tokendict[' ']=27
  def __len__(self):
    return len(self.tokendict.keys)
  def tokenize(self,text):
    return [self.tokendict[i] for i in text]


    


class ENCODER(Module):
  def __init__(self,config):
    super().__init__()

    self.embedding=Embedding(config.num_tokens, config.enc_dim)
    self.rnn= GRU(config.enc_dim, config.enc_dim//2,bidirectional=True,batch_first=True)
    self.fc1=Linear(config.enc_dim*2 ,config.enc_dim)
  
  def forward(self,X,lens):


    y=self.embedding(X)


    y = pack_padded_sequence(y,lens , batch_first=True,enforce_sorted=False)
    self.rnn.flatten_parameters()
    output, state = self.rnn(y)
    output, _ = pad_packed_sequence(output, batch_first=True, padding_value=0.)

    return output, state

class Location_awarness(Module):
  def __init__(self,config):
    super().__init__()

    padding = int((config.attn_kernel - 1) / 2)
    self.Conv = Conv1d(2, config.attn_filter,config.attn_kernel ,padding=padding,bias=False)

    # fully connected layer to convert the features extracted from [cummulative,prev] for encoderop len  to features of attention dimension
    self.fc   = Linear(config.attn_filter,config.attn_dim,bias=False)

  def forward(self,stacked_probs):

    y=self.Conv(stacked_probs)
    y=torch.transpose(y,1,2)
    y=self.fc(y)

    return y


class Attention(Module):
  def __init__(self,config):
    super().__init__()

    self.W1=Linear(config.enc_dim,config.attn_dim,bias=False)
    self.W2=Linear(config.dec_dim,config.attn_dim,bias=False)
    self.V =Linear(config.attn_dim,1,bias=False)

    self.location=Location_awarness(config)

  def forward(self,prev_hidden,encoder_op,prev_attn,mask):


    eo=self.W1(encoder_op)
    ph=self.W2(prev_hidden.unsqueeze(1))

    pa=self.location(prev_attn)

    scores=self.V(torch.tanh(eo+ph+pa))


    scores.data.masked_fill_(mask.unsqueeze(-1) == 0, -1e10)
    #print(scores.detach().tolist())
    #print((mask.unsqueeze(-1) == 0).detach().tolist())

    attn_probs= F.softmax(scores, dim = 1)

    context_vector = attn_probs * encoder_op
    context_vector = torch.sum(context_vector, dim=1)

    return context_vector,attn_probs.squeeze(-1)

class PRENET(Module):
    def __init__(self, config):
        super().__init__()
        self.layers = ModuleList(
            [Linear(config.spectrogram_dimension,config.prenet_dim,bias=False),
             Linear(config.prenet_dim,config.prenet_dim,bias=False)]
            )

    def forward(self, x):
        for linear in self.layers:
            x = F.dropout(F.relu(linear(x)), p=0.5, training=True)
        return x
class DECODER(Module):
  def __init__(self,config):
    super().__init__()

    self.prenet=PRENET(config)
    self.rnn1 = GRU(config.prenet_dim + config.contextvec_dim, config.dec_dim,batch_first=True)
    self.rnn2 = GRU(config.dec_dim + config.contextvec_dim, config.dec_dim,batch_first=True)
    self.attention=Attention(config)

    self.spec_pred=Linear(config.dec_dim+config.contextvec_dim,config.spectrogram_dimension,bias=False)
    self.stop_pred=Linear(config.dec_dim+config.contextvec_dim,1,bias=False) # stop is binary

    self.bridge=Linear(config.enc_dim,config.dec_dim,bias=False)

    self.drop1=Dropout(0.1)
    self.drop2=Dropout(0.1)
    

    self.nst=config.spectrogram_dimension

    

  def compute_step(self,curr_ip,prev_hidden,encoder_op,mask):

    prev_attn=torch.stack([self.prev_attn_probs,self.prev_Atn_cum],dim=1)
    context_vector,self.prev_attn_probs=self.attention(prev_hidden,encoder_op,prev_attn,mask)
    self.prev_Atn_cum=self.prev_Atn_cum+self.prev_attn_probs


    y=self.prenet(curr_ip)


    y=torch.cat([y,context_vector],dim=-1)
    y=torch.unsqueeze(y,1) # making seq of len 1
    y,_=self.rnn1(y)
    y=self.drop1(y)
    
    y=torch.cat([y,context_vector.unsqueeze(1)],dim=-1)
    y,state=self.rnn2(y,prev_hidden.unsqueeze(0))
    y=self.drop2(y)

    y=torch.squeeze(y,1) # making seq of len 1 to a vector of tht time step

    y=torch.cat([y,context_vector],dim=-1)

    spec=self.spec_pred(y)
    stop=self.stop_pred(y)

    return spec,state,stop.squeeze(-1)


  def start_token(self,B):
    return  Variable(torch.zeros(B,self.nst))

  def forward(self,encoder_ops,decoder_ip,oldencoder_mask):

    max_len=decoder_ip.shape[1]
    encoder_op,state=encoder_ops
    B=encoder_op.shape[0]
    curr_ip=self.start_token(B).to(encoder_op.device)
    hidden=self.bridge(torch.cat([state[0],state[1]],-1))

    spectrogram_op=[]
    stop_token_op=[]

    encoder_mask=oldencoder_mask[:,:encoder_op.shape[1]]


    self.prev_Atn_cum=Variable(encoder_op.data.new(B, encoder_op.shape[1]).zero_())
    self.prev_attn_probs=Variable(encoder_op.data.new(B, encoder_op.shape[1]).zero_()) 

    for t in range(max_len):




          predictions, hidden,stop_pred = self.compute_step(curr_ip,hidden,encoder_op,encoder_mask)



          spectrogram_op.append(predictions)
          stop_token_op.append(torch.sigmoid(stop_pred))
          curr_ip = decoder_ip[:, t]
          #if(t in [0,1,2,5,10,20,50,100,200,500]):print(F.mse_loss(predictions,curr_ip).item(),t)

          hidden=hidden.squeeze(0)

          

    spectrogram_op=torch.stack(spectrogram_op).transpose(0,1)
    stop_token_op=torch.stack(stop_token_op).transpose(0,1)
    return spectrogram_op,stop_token_op

  def inference(self,encoder_ops,oldencoder_mask):

    max_len=814
    encoder_op,state=encoder_ops
    B=encoder_op.shape[0]
    curr_ip=self.start_token(B).to(encoder_op.device)
    hidden=self.bridge(torch.cat([state[0],state[1]],-1))

    spectrogram_op=[]
    stop_token_op=[]


    encoder_mask=oldencoder_mask[:,:encoder_op.shape[1]]

    self.prev_Atn_cum=Variable(encoder_op.data.new(B, encoder_op.shape[1]).zero_())
    self.prev_attn_probs=Variable(encoder_op.data.new(B, encoder_op.shape[1]).zero_()) 
    for t in range(max_len):

          predictions, hidden,stop_pred = self.compute_step(curr_ip,hidden,encoder_op,encoder_mask)
          
          xy=torch.sigmoid(stop_pred)
          #print(xy)
          if(xy<0.98):
              break
          spectrogram_op.append(predictions)

          curr_ip = predictions

          hidden=hidden.squeeze(0)
          
    spectrogram_op=torch.stack(spectrogram_op).transpose(0,1)
    return spectrogram_op,stop_token_op




class Resnet_block(Module):
  def __init__(self,indim,outdim,kernel):

    super().__init__()

    self.conv1 = Conv1d(indim,outdim,kernel,padding=(kernel-1)//2,bias=False)
    self.bn1   = BatchNorm1d(outdim)

    self.conv2 = Conv1d(outdim,outdim,kernel,padding=(kernel-1)//2,bias=False)
    self.bn2   = BatchNorm1d(outdim)

    self.skip=Sequential(
        Conv1d(indim,outdim,1,bias=False),
        BatchNorm1d(outdim)
    )

  def forward(self,X):

    y= self.bn1(self.conv1(X))
    y= self.bn2(self.conv2(y))  +self.skip(X)

    return torch.tanh(y)

class Enhancer(Module):
  def __init__(self,config):
    super().__init__()

    self.blocks=ModuleList()
    self.fc=ModuleList()

    num_layers=4

    for i in range(num_layers):

      ind=config.spectrogram_dimension if i==0 else config.enchancer_dim
      outd=config.enchancer_dim if i< num_layers-1 else config.spectrogram_dimension

      self.blocks.append(Resnet_block(ind,outd,5))
      self.fc.append(Linear(outd,outd,bias=False))

    self.avgpool = AdaptiveAvgPool1d(1)
    self.N=num_layers
    self.loss=MSELoss()

  def forward(self,X,training=True):

    y=X.transpose(1,2)



    for i in range(self.N):

      y=F.dropout(self.blocks[i](y),0.25)
     
    return y.transpose(1,2)


class EDAttn(Module):
  def __init__(self,config):
    super().__init__()
    self.encoder=ENCODER(config)
    self.decoder=DECODER(config)
    self.enhancer=Enhancer(config)
    self.loss=MSELoss(reduction='none')
    self.lossstop=MSELoss()

  def loss_spectrogram(self,prediction,target,mask) : #summed mean squared error (MSE) from before and after the post-net to aid convergence
       
       a,b,stop=prediction
       spec_target,stop_target=target

       stop_loss=self.lossstop(stop,mask)
     
       mask=mask.unsqueeze(-1).expand_as(a)


       aloss=(self.loss(a,spec_target)* mask.float()).sum()/  mask.sum()
       bloss=(self.loss(b,spec_target)* mask.float()).sum()/  mask.sum() 


    
       return [aloss,bloss,stop_loss]
         
 
  def forward(self,input,decoder_input,mask,lens):

    y=self.encoder(input,lens)

    spec,stop=self.decoder(y,decoder_input,mask)
    spech_enh=self.enhancer(spec)+spec

    return spec,spech_enh,stop

  def inference(self,input,mask,lens):
    y=self.encoder(input,lens)
    y,stop=self.decoder.inference(y,mask)
    y1=self.enhancer(y)+y
    return y,y1,stop




def avg(x):
  return sum(x)/len(x)


class pipeline():

    def __init__(self,model,processing_unit):
        
     self.device = torch.device(processing_unit)
     self.model=model
     self.model.to(self.device)
    

    def loadmodel(self,path):
      self.model.load_state_dict(torch.load(path,map_location=self.device))  


    def predict(self,text,vocoder=None):
     
     self.model.eval()
     cleaned_sentence = re.findall(r'[A-Za-z]+|\d+', clean(text))   
     cleanedtext=' '.join([num2words(i).replace(',','').replace('-',' ') if i.isnumeric() else i for i in cleaned_sentence])
     tokenized_text=char_tokenizer().tokenize(cleanedtext)

     ln=len(tokenized_text)
     mask=np.ones(ln)
     data={'text':torch.tensor(tokenized_text),
             'len':torch.tensor(ln,dtype=torch.int32),
             'mask':torch.tensor(mask,dtype=torch.int32),}



     for key,value in data.items():
              data[key]=value.unsqueeze(0)

     inp,mask,enc_len=data['text'],data['mask'],data['len']

     with torch.no_grad():
         spec_op,spec_enh,stop=self.model.inference(inp,mask,enc_len)    

    


     if(vocoder):
       with torch.no_grad():
            audio = vocoder.inference(spec_enh.transpose(1,2))
            
            return audio.detach().numpy()



    
class cfg():
  def __init__(self,no_tokens):

    #default
    self.enc_dim=512 # conventional encoder
    self.dec_dim=1024 # 1024 worked better than 768
    self.attn_dim=128  # 256 cuda overflow

    self.contextvec_dim=self.enc_dim # adding encoder op on timesteps weighted scores will result vecotor of encder dimension

    #encoder
    self.num_tokens=no_tokens
  

    # attn location awarness (same as location awareness paper)
    self.attn_filter=32
    self.attn_kernel=31

    #spectrogram
    self.spectrogram_dimension=80 # number of mel channels
    self.no_framesperstep=1
    self.spectrogram_dimension*=self.no_framesperstep
    # number of frames to be predicted is 1  because it gives good results
    
    # prenet 
    self.prenet_dim=64 * self.no_framesperstep
    # checking 64 ,128,256      64 didn't have overfitting while others had

    # postnet
    self.enchancer_dim=256

class synthesiser:
    def __init__(self,path):
        
        component=EDAttn(cfg(28))
        
        self.Model=pipeline(component,'cpu')
        self.Model.loadmodel(path+"text2spectrebuild.pth")
        
        self.vocoder = torch.hub.load(path , 'melgan')
        self.vocoder.to(torch.device('cpu'))
        self.vocoder.eval()
    def __call__(self,text):
        
        curr=""
        audio=np.ones((10))*15956
        for word in text.split():
            curr+=word
            if(len(curr)>200):
                seg=self.Model.predict(curr,self.vocoder)
                audio=np.concatenate([audio,seg])
                audio=np.concatenate([audio,np.ones((20))*15956])
                
                curr=''
                print(len(audio))
                
        #aud= self.Model.predict("kamalesh vignesh mitran nitin harsha digest audio casette",self.vocoder)
        if(len(curr)):
            seg=self.Model.predict(curr,self.vocoder)
            audio=np.concatenate([audio,seg])
        
        return audio
        
        
"""     
tts = synthesiser('./TTSweights/')
text="kamalesh vignesh mitran nitin harsha digest audio casette"

x=tts(text)

import scipy
m = np.max(np.abs(x))
sigf32 = (x/m).astype(np.float32)
scipy.io.wavfile.write("check.mp3", 22050, sigf32)"""


   