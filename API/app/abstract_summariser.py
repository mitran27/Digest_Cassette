import tensorflow as tf
import numpy as np
import pandas as pd
from functools import lru_cache
import pickle

from tensorflow.keras.models import Model
from tensorflow.keras.layers  import Layer,LayerNormalization,Embedding,Dropout
from tensorflow.keras.layers  import Dense as Linear
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.activations import softmax
from tensorflow.keras.optimizers.schedules import LearningRateSchedule
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy 



def Positional_Encoding(length,model_dimension):
  
  md=model_dimension
  array=np.zeros((length,md))
  for pos in range(length):
    for i in range(md):
      if (i%2==0):
        array[pos][i]=np.sin(pos/np.power(10000,i/np.float32(md)))
      else:
        array[pos][i]=np.cos(pos/np.power(10000,(i-1)/np.float32(md)))

  return tf.cast(array ,dtype=tf.float32)


@lru_cache(None)
def masking(seq_len):
   return 1-tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)

class Attention(Layer):
    
      def __init__(self,dimension,num_heads):
    
        super().__init__()
    
        assert(dimension%num_heads==0 )
        self.num_heads=num_heads
        self.vec_dim=dimension//num_heads 
        self.scaling = tf.math.sqrt(tf.cast(self.vec_dim,dtype=tf.float32))
    
        # projection are the weights
       
        self.q_proj =Linear(dimension)
        self.k_proj =Linear(dimension)
        self.v_proj =Linear(dimension)
    
        self.out_proj=Linear(dimension)
    
      def split_heads(self,tensor):
    
        bz,seqlen,dimension=tensor.shape
        assert(self.vec_dim*self.num_heads==dimension)
        tensor=tf.reshape(tensor,(bz,seqlen,self.num_heads,self.vec_dim))
        tensor=tf.transpose(tensor,perm=[0,2,1,3])
        tensor=tf.reshape(tensor,(bz*self.num_heads,seqlen,self.vec_dim))
    
        return tensor
    
      def concat_heads(self,tensor):
    
        bz_x_nohead,seqlen,vec_dim=tensor.shape
        assert(vec_dim==self.vec_dim)
        assert(bz_x_nohead%self.num_heads==0)
        bz= bz_x_nohead//self.num_heads
        tensor=tf.reshape(tensor,(bz,self.num_heads,seqlen,vec_dim))
        tensor=tf.transpose(tensor,perm=[0,2,1,3])
        tensor=tf.reshape(tensor,(bz,seqlen,self.num_heads*vec_dim))
        return tensor
      # writing code in common for all varieties of attention =>(encoder,decoder,masked)  
      def call(self,query,key,value,mask=False):
    
    
        query_states=self.q_proj(query)
        
        key_states = self.k_proj(key)
        value_states = self.v_proj(value)
    
        query_states = self.split_heads(query_states)
        key_states =   self.split_heads(key_states)
        value_states = self.split_heads(value_states)
    
        # scaled dot product attention
        #print(query_states.shape,tf.transpose(key_states,perm=[0,2,1]).shape)
        attn_weights = tf.matmul(query_states,tf.transpose(key_states,perm=[0,2,1]))/self.scaling
    
        if(mask):
          seqlen=query.shape[-2]
          attn_weights += ( masking(seqlen)*-1e9 )
     
        attn_probs = softmax(attn_weights, axis=-1)  
        attn_out = tf.matmul(attn_probs,value_states)
        attn_out = self.concat_heads(attn_out)
    
        attn_output = self.out_proj(attn_out)
        
        return attn_output





class Feed_Forward(Layer):
  def __init__(self,dimension,intermediate_size):
    super().__init__()
   
    self.intermediate_dense=Linear(intermediate_size,activation='relu')
    self.output_dense=Linear(dimension)

  def call(self,X):

    y=self.intermediate_dense(X)
    y=self.output_dense(y)
    
    return y



class Encoder_block(Layer):
  def __init__(self,config):

    super().__init__()
    self.multiheadattn=Attention(config.dimension,config.no_heads)
    self.FFn=Feed_Forward(config.dimension,config.intermediate_size)

    self.layer_norm1 = LayerNormalization(epsilon=config.normeps)
    self.layer_norm2 = LayerNormalization(epsilon=config.normeps)

  def call(self,X,training=True):

    y=self.multiheadattn(X,X,X)
    y=self.layer_norm1(X+y)

    y1=self.FFn(y)
    y1=self.layer_norm2(y+y1)

    return y1


class Decoder_block(Layer):
  def __init__(self,config):

    super().__init__()

    self.multiheadattn=Attention(config.dimension,config.no_heads)
    self.maskedattn=Attention(config.dimension,config.no_heads)
    self.FFn=Feed_Forward(config.dimension,config.intermediate_size)

    self.layer_norm1 = LayerNormalization(epsilon=config.normeps)
    self.layer_norm2 = LayerNormalization(epsilon=config.normeps)
    self.layer_norm3 = LayerNormalization(epsilon=config.normeps)

  def call(self,X,encoder_op,training=True):

    y=self.maskedattn(X,X,X,mask=True)
    y=self.layer_norm1(X+y)

    y1=self.multiheadattn(y,encoder_op,encoder_op)
    y1=self.layer_norm2(y+y1)

    y2=self.FFn(y1)
    y2=self.layer_norm3(y2+y1)

    return y2



class Encoder(Layer):

  def __init__(self,config,enc_cfg):
    super().__init__()

    self.num_layers=config.no_layers
    self.emb_scale=tf.cast(config.dimension,dtype=tf.float32)
    self.embedding=Embedding(enc_cfg['vocab_Size'],config.dimension)
    self.pos_enc=Positional_Encoding(enc_cfg['max_len'],config.dimension)
    self.pos_enc=tf.expand_dims(self.pos_enc,axis=0)
    self.no_layers=config.no_layers
    self.encoder_blocks = [Encoder_block(config) for _ in range(config.no_layers)]

    self.drop=Dropout(config.dropout)

  def call(self,X,training=True):
    y=self.embedding(X)
    y*=tf.math.sqrt(self.emb_scale)
    seqlen=X.shape[-1]
    y+=self.pos_enc[:,:seqlen,:]
    y=self.drop(y,training=training)
    for i in range(self.no_layers):
      y=self.encoder_blocks[i](y,training)
    
    return y  


class Decoder(Layer):

  def __init__(self,config,dec_cfg):
    super().__init__()

    self.num_layers=config.no_layers
    self.emb_scale=tf.cast(config.dimension,dtype=tf.float32)
    self.embedding=Embedding(dec_cfg['vocab_Size'],config.dimension)
    self.pos_enc=Positional_Encoding(dec_cfg['max_len'],config.dimension)
    self.pos_enc=tf.expand_dims(self.pos_enc,axis=0)
    self.no_layers=config.no_layers

    self.decoder_blocks = [Decoder_block(config) for _ in range(config.no_layers)]

    self.drop=Dropout(config.dropout)

  def call(self,X,encoder_op,training=True):
    y=self.embedding(X)
    y*=tf.math.sqrt(self.emb_scale)
    seqlen=X.shape[-1]
    y+=self.pos_enc[:,:seqlen,:]
    y=self.drop(y,training=training)
    for i in range(self.no_layers):
      y=self.decoder_blocks[i](y,encoder_op,training)
    return y  

class Transfomer(Layer):

  def __init__(self,config,enc_cfg,dec_cfg):
    super().__init__()

    self.encoder=Encoder(config,enc_cfg)
    self.decoder=Decoder(config,dec_cfg)
    
    self.drop=Dropout(config.dropout)
    self.final=Linear(dec_cfg['vocab_Size'])

    self.lossfn=SparseCategoricalCrossentropy(from_logits=True, reduction='none')
  def loss(self,predicted,truth):

    loss_ = self.lossfn(truth, predicted)

    mask=tf.math.logical_not(tf.math.equal(truth, 0))
    mask_float = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask_float

    accuracy = tf.equal(truth, tf.argmax(predicted, axis=-1))
    #print(accuracy.shape,mask.shape,mask.dtype,accuracy.dtype)
    accuracy=tf.math.logical_and(mask, accuracy)
    accuracy = tf.cast(accuracy, dtype=mask_float.dtype)


    return tf.reduce_sum(loss_)/tf.reduce_sum(mask_float),tf.reduce_sum(accuracy)/tf.reduce_sum(mask_float)

  def call(self,source,target,training=True):
    enc_op=self.encoder(source,training)
    dec_op=self.decoder(target,enc_op,training)
    op=self.drop(dec_op,training=training)
    op=self.final(op)

    return op




class Pipeline():
  def __init__(self,component):

    self.model=component


  def load_model(self,path):
    ckpt = tf.train.Checkpoint(model=self.model)
    ckpt_manager = tf.train.CheckpointManager(ckpt, path, max_to_keep=5)
    if ckpt_manager.latest_checkpoint:
          ckpt.restore(ckpt_manager.latest_checkpoint)
          print('Latest checkpoint restored!!')
          
  def predict(self,sentence,source_tokenizer,target_tokenizer):
    tokenized=np.array((source_tokenizer.texts_to_sequences([sentence])))
    output_sentence="summstart"
     
    while(True):
      tensor=target_tokenizer.texts_to_sequences([output_sentence])
      tensor=tf.cast(np.array(tensor*1),dtype=tf.int64)
      z=self.model(tokenized,tensor,training=False)
      
      z=tf.argmax(z[0],axis=-1)
      lastword=z.numpy()[-1]
      output_sentence+=' '+target_tokenizer.index_word[lastword]
      
      if(lastword==target_tokenizer.word_index['summend'] or len(output_sentence.split())>100):return output_sentence


class config():
  def __init__(self):
    self.dimension=128
    self.no_heads=8
    self.no_layers=4
    self.intermediate_size=self.dimension*4
    self.normeps= int(1e-6)
    self.dropout=0.1
    
    

    





  







class Abstract_Summ:

    def __init__(self,path):

        Transfomerconfig=config()
        
        
        
        
                    
        
        
        
        with open(path+'sourcetokenizer.pickle', 'rb') as handle:
            self.source_tokenizer = pickle.load(handle)
        
        with open(path+'targettokenizer.pickle', 'rb') as handle:
            self.target_tokenizer = pickle.load(handle)
        
        
        vocab_size_source=len(self.source_tokenizer.word_index)+1 # one for the padding of zeros
        vocab_size_target=len(self.target_tokenizer.word_index)+1
        
        encoder_config={"vocab_Size":vocab_size_source,"max_len":100}
        decoder_config={"vocab_Size":vocab_size_target,"max_len":100}



        component=Transfomer(Transfomerconfig,encoder_config,decoder_config)
        self.Model=Pipeline(component)
        self.Model.load_model(path+"checkpoint/")
    def summarize(self,sentence):
        
        x= self.Model.predict(sentence,self.source_tokenizer,self.target_tokenizer)
        return x.replace("summstart","").replace("summend","")

"""
dn="summstart the paradise papers leak has revealed how funds amounting to over 1.5 billion were allegedly diverted using four offshore subsidiaries of united spirits limited india , when vijay mallya owned the company. after diageo acquired usl in 2013, it undertook a restructuring process to get rid of three of the subsidiaries and ended up waiving debts the firms owed. summend "

Summarizer = Abstract_Summ('./transformerweights/')

text=Summarizer.summarize(dn)"""


