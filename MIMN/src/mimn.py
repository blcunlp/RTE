################
#20170918
# implement of enhanced LSTM, small modify
# the input is deliver by feed_dict
################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import time
import inspect
import logging
import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn.python.ops import core_rnn_cell
from tensorflow.contrib.layers import batch_norm,l2_regularizer
from tensorflow.python.ops import variable_scope

from ops_cudnn_rnn import *

class MyModel(object):
  """The MIMN model."""

  def __init__(self,pretrain_embedding, is_training, config):

    batch_size = config.batch_size
    self.config = config
    self.is_training = is_training
    self.global_step = tf.Variable(0, trainable=False)
    self.matching_order = config.matching_order

    self.embedding = pretrain_embedding

    self.learning_rate = tf.Variable(self.config.learning_rate, trainable=False)
    self.add_placeholder()
    with tf.variable_scope("add-embedding") as scope:
      input_xemb,input_yemb = self.add_pretrained_embedding()
      input_xemb = input_xemb*self.x_mask[:,:,None]
      input_yemb = input_yemb*self.y_mask[:,:,None]

    with tf.variable_scope("encode_x") as scope:
      self.x_output = cudnn_lstm(inputs=input_xemb,
                                 num_layers=1,
                                 hidden_size=self.config.hidden_units,
                                 is_training=self.is_training,
                                 regularizer=l2_regularizer(self.config.l2_strength))    

      scope.reuse_variables()
      self.y_output= cudnn_lstm(inputs=input_yemb,num_layers=1,hidden_size=self.config.hidden_units,
                                is_training=self.is_training,regularizer=l2_regularizer(self.config.l2_strength))    

      if is_training and config.keep_prob < 1:
        self.x_output = tf.nn.dropout(self.x_output, config.keep_prob)
        self.y_output = tf.nn.dropout(self.y_output, config.keep_prob)

      self.x_output = self.x_output*self.x_mask[:,:,None]
      self.y_output = self.y_output*self.y_mask[:,:,None]
    
    with tf.variable_scope("dot-product-atten") :
      #weightd_y:(b,x_len,2*h),weighted_x:(b,y_len,2*h)
      self.weighted_y, self.weighted_x =self.dot_product_attention(x_sen=self.x_output,
                                                                   y_sen=self.y_output,
                                                                   x_mask = self.x_mask,
                                                                   y_mask = self.y_mask)

    with tf.variable_scope("aggregate") as scope:
      mxy_output = self.multi_turn_inference(x=self.x_output, att_y =self.weighted_y) 
      scope.reuse_variables()
      myx_output = self.multi_turn_inference(x=self.y_output, att_y =self.weighted_x) 
      
    with tf.variable_scope("pooling"):
      v_xymax = tf.reduce_max(mxy_output,axis=1)  #(b,2h)    
      v_xy_sum = tf.reduce_sum(mxy_output, 1)  #(b,x_len.2*h) ->(b,2*h)
      v_xyave = tf.div(v_xy_sum, tf.expand_dims(self.x_len, -1)) #div true length

      v_yxmax = tf.reduce_max(myx_output,axis=1)  #(b,2h)
      v_yx_sum = tf.reduce_sum(myx_output, 1)   ##(b,y_len.2*h) ->(b,2*h)
      v_yxave = tf.div(v_yx_sum, tf.expand_dims(self.y_len,  -1)) #div true length

      self.v = tf.concat([v_xyave,v_xymax,v_yxmax,v_yxave],axis=-1) #(b,8*h)
      if is_training and config.keep_prob < 1:
        self.v = tf.nn.dropout(self.v, config.keep_prob)

    with tf.variable_scope("pred-layer"):
      fnn1 = self.fnn(input=self.v,
                      out_dim=self.config.hidden_units,
                      activation=tf.nn.tanh,
                      use_bias=True,
                      w_name="fnn-pred-W")
  
      if is_training and config.keep_prob < 1:
        fnn1 = tf.nn.dropout(fnn1, config.keep_prob)

      W_pred = tf.get_variable("W_pred", shape=[self.config.hidden_units, self.config.num_classes],regularizer=l2_regularizer(self.config.l2_strength))
      self.pred = tf.nn.softmax(tf.matmul(fnn1, W_pred), name="pred")

      correct = tf.equal(tf.argmax(self.pred,1),tf.argmax(self.label,1))
      self.acc = tf.reduce_mean(tf.cast(correct, "float"), name="accuracy")
    
      self.loss_term = -tf.reduce_sum(tf.cast(self.label,tf.float32) * tf.log(self.pred),name="loss_term")
      self.reg_term = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES),name="reg_term")
      
      va = tf.concat([v_xyave,v_xymax],axis=-1) #(b,8*h)
      vb = tf.concat([v_yxmax,v_yxave],axis=-1) #(b,8*h)

      self.loss = tf.add(self.loss_term,self.reg_term,name="loss")

    if not is_training:
        return
    with tf.variable_scope("bp_layer"):
      tvars = tf.trainable_variables()
      print ("-"*45)
      print ("trainable parameters:")
      for var in tvars:
        print (var.name,var.shape)
      print ("-"*45)
      grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars),
                                      config.max_grad_norm)
      optimizer = tf.train.AdamOptimizer(self.learning_rate)
      self.optim = optimizer.apply_gradients(
          zip(grads, tvars),
          global_step=self.global_step)

  def add_placeholder(self):
    
    self.x = tf.placeholder(tf.int32, [self.config.batch_size, self.config.xmaxlen],name="x")
    self.y = tf.placeholder(tf.int32, [self.config.batch_size, self.config.ymaxlen],name="y")

    self.x_mask = tf.placeholder(tf.float32, [self.config.batch_size, self.config.xmaxlen],name="x_mask")
    self.y_mask = tf.placeholder(tf.float32, [self.config.batch_size, self.config.ymaxlen],name="y_mask")

    self.x_len = tf.placeholder(tf.float32, [self.config.batch_size,],name="x_len")
    self.y_len = tf.placeholder(tf.float32, [self.config.batch_size,],name="y_len")

    self.label = tf.placeholder(tf.int32, [self.config.batch_size,self.config.num_classes],name="label")
    self._new_lr = tf.placeholder(tf.float32, shape=[], name="new_learning_rate")
    self._lr_update = tf.assign(self.learning_rate, self._new_lr)


  def add_pretrained_embedding(self):

    with tf.device("/cpu:0"):
      x_glove_emb = tf.nn.embedding_lookup(self.embedding, self.x)
      y_glove_emb = tf.nn.embedding_lookup(self.embedding, self.y)
    
      if self.is_training and self.config.keep_prob < 1:
        x_glove_emb = tf.nn.dropout(x_glove_emb, self.config.keep_prob)
        y_glove_emb = tf.nn.dropout(y_glove_emb, self.config.keep_prob)
    return x_glove_emb,y_glove_emb

  def add_pos_embedding(self):
  
    with tf.device("/cpu:0"):
      #pos_embedding = tf.Variable(tf.random_normal([self.config.pos_num,self.config.pos_embedding_dim]),trainable=True, name="pos_embedding")
      pos_embedding = tf.get_variable(name="pos_embedding",shape=[self.config.pos_num,self.config.pos_embedding_dim],trainable=True) 
      x_pos_emb = tf.nn.embedding_lookup(pos_embedding, self.x_pos)
      y_pos_emb = tf.nn.embedding_lookup(pos_embedding, self.y_pos)
    
      if self.is_training and self.config.keep_prob < 1:
        x_pos_emb= tf.nn.dropout(x_pos_emb, self.config.keep_prob)
        y_pos_emb = tf.nn.dropout(y_pos_emb, self.config.keep_prob)
    return x_pos_emb,y_pos_emb
  
  def add_char_embedding(self):
    with tf.variable_scope("char_emb"):
      char_emb_mat = tf.get_variable("char_emb_mat", shape=[self.config.char_vocab_size, self.config.char_emb_size])
      with tf.variable_scope("char") as scope:
        char_x = tf.nn.embedding_lookup(char_emb_mat, self.x_char)
        char_y = tf.nn.embedding_lookup(char_emb_mat, self.y_char)

        filter_sizes = list(map(int, self.config.out_channel_dims.split(','))) #[100]
        heights = list(map(int, self.config.filter_heights.split(',')))        #[5]
        assert sum(filter_sizes) == self.config.char_out_size, (filter_sizes, self.config.char_out_size)
        with tf.variable_scope("conv") as scope:
          conv_x = multi_conv1d(char_x, filter_sizes, heights, "VALID", tf.cast(self.is_training,tf.bool), self.config.keep_rate, scope='conv')
          scope.reuse_variables()  
          conv_y = multi_conv1d(char_y, filter_sizes, heights, "VALID", tf.cast(self.is_training,tf.bool), self.config.keep_rate, scope='conv')
          conv_x = tf.reshape(conv_x, [-1, self.config.xmaxlen, self.config.char_out_size])
          conv_y = tf.reshape(conv_y, [-1, self.config.ymaxlen, self.config.char_out_size])
    return conv_x,conv_y

  def dot_product_attention(self,x_sen,y_sen,x_mask,y_mask):
    '''
    function: use the dot-production of left_sen and right_sen to compute the attention weight matrix
    :param left_sen: a list of 2D tensor (x_len,hidden_units)
    :param right_sen: a list of 2D tensor (y_len,hidden_units)
    :return: (1) weighted_y: the weightd sum of y_sen, a 3D tensor with shape (b,x_len,2*h)
             (2)weghted_x:  the weighted sum of x_sen, a 3D tensor with shape (b,y_len,2*h)
    '''
    
    weight_matrix =tf.matmul(x_sen, tf.transpose(y_sen,perm=[0,2,1])) #(b,x_len,2h) x (b,2h,y_len)->(b,x_len,y_len)

    weight_matrix_y =tf.exp(weight_matrix - tf.reduce_max(weight_matrix,axis=2,keep_dims=True))  #(b,x_len,y_len)
    weight_matrix_x =tf.exp(tf.transpose((weight_matrix - tf.reduce_max(weight_matrix,axis=1,keep_dims=True)),perm=[0,2,1]))  #(b,y_len,x_len)

    weight_matrix_y=weight_matrix_y*y_mask[:,None,:]#(b,x_len,y_len)*(b,1,y_len)
    weight_matrix_x=weight_matrix_x*x_mask[:,None,:]#(b,y_len,x_len)*(b,1,x_len)
    
    alpha=weight_matrix_y/(tf.reduce_sum(weight_matrix_y,-1,keep_dims=True)+1e-8)#(b,x_len,y_len)
    beta=weight_matrix_x/(tf.reduce_sum(weight_matrix_x,-1,keep_dims=True)+1e-8)#(b,y_len,x_len)

    #(b,1,y_len,2*h)*(b,x_len,y_len,1)*=>(b,x_len,y_len,2*h) =>(b,x_len,2*h)
    weighted_y =tf.reduce_sum(tf.expand_dims(y_sen,1) *tf.expand_dims(alpha,-1),2)

    #(b,1,x_len,2*h)*(b,y_len,x_len,1) =>(b,y_len,x_len,2*h) =>(b,y_len,2*h)
    weighted_x =tf.reduce_sum(tf.expand_dims(x_sen,1) * tf.expand_dims(beta,-1),2)

    return weighted_y,weighted_x

  def tensordot(self,inp,out_dim,in_dim=None,activation=None,use_bias=False,w_name="batch-fnn-W"):
    '''
    function: the implement of FNN ,input is a 3D batch tesor,W is a 2D tensor
    :param input: a 3D tensor of (b,seq_len,h)
    :param out_dim: the out_dim of W
    :param in_dim: the in_dim of W
    :param activation: activation function
    :param use_bias: use bias or not
    :param w_name: the unique name for W
    :return: (b,seq_len,in_dim)*(in_dim,out_dim) ->(b,seq_len,out_dim)
    '''
    with tf.variable_scope("3D-batch-fnn-layer"):
      inp_shape = inp.get_shape().as_list()
      batch_size= inp_shape[0]
      seq_len = inp_shape[1]

      if in_dim==None:
        in_dim = inp_shape[-1]
      
      W = self.get_variable(w_name,shape=[in_dim,out_dim])
      out = tf.tensordot(inp,W,axes=1)

      if use_bias == True:
        b_name = w_name + '-b'
        b = tf.get_variable(b_name, shape=[out_dim])
        out = out + b

      if activation is not None:
        out = activation(out)
      out.set_shape([batch_size,seq_len,out_dim])
      return out


  def get_h_n(self,lstm_y_output ,true_y_length):
    ''' lstm_y_output: A 3D Tensor of shape(sey_len, batch_size,hidden_units)(list will work)
      true_y_length:(batch_size)
    '''
    hn_list=[]
    for i in range(self.config.batch_size):
      i=tf.cast(i,tf.int32)

      last_step=tf.cast(true_y_length[i]-1,tf.int32)

      hn_list.append(lstm_y_output[last_step, i, :])
      
    hn_tensor=tf.convert_to_tensor(hn_list)
    h_n=tf.reshape(hn_tensor,[tf.shape(hn_tensor)[0], tf.shape(hn_tensor)[-1]])         
    return h_n
    


  def fnn(self,input,out_dim,in_dim=None,activation=None,use_bias=False,w_name="fnn-W"):
    with tf.variable_scope("fnn-layer"):
      if in_dim==None:
        input_shape = input.get_shape().as_list()
        in_dim = input_shape[-1]
      
      W = tf.get_variable(w_name,shape=[in_dim,out_dim])
      out = tf.matmul(input,W)

      if use_bias == True:
        b_name = w_name + '-b'
        b = tf.get_variable(b_name, shape=[out_dim])
        out = out + b

      if activation is not None:
        out = activation(out)
    return out


  def assign_lr(self, session, lr_value):
    session.run(self._lr_update, feed_dict={self._new_lr: lr_value})
  
  def multi_turn_inference(self,x,att_y):
    shape= x.get_shape().as_list()
    hidden_units= shape[-1]
    seq_length =shape[1]

    def update_memory(x,prev_mem,epoch,use_gated=False):
      
      with tf.variable_scope("update_memory") as scope:
        if epoch>0:
          scope.reuse_variables()
        
        inp = tf.concat([x,prev_mem],axis=-1) 
        if use_gated==True: 
          #W_g = self.get_variable("W_g", shape=[inp.get_shape().as_list()[-1],inp.get_shape().as_list()[-1]])
          gt = tf.nn.sigmoid(self.tensordot(inp,out_dim=inp.get_shape().as_list()[-1], w_name="gt") ) 
          inp = gt*inp
       
        inp = self.tensordot(inp=inp,out_dim=self.config.hidden_units, w_name="inp_w") 
        lstm_output= cudnn_lstm(inputs=inp,num_layers=1,
                                hidden_size=self.config.hidden_units,
                                is_training=self.is_training,
                                regularizer=l2_regularizer(self.config.l2_strength))    
        output_gate = tf.nn.sigmoid(self.tensordot(tf.concat([lstm_output,prev_mem],axis=-1),out_dim=lstm_output.get_shape().as_list()[-1], w_name="gate") ) 
        output = output_gate*lstm_output + (1-output_gate)*prev_mem
      return output

    #compare_feature =[tf.concat([x,att_y],axis=-1),x-att_y,x*att_y]
    #compare_feature =[x-att_y,x*att_y,tf.concat([x,att_y],axis=-1)]
    cat = tf.concat([x,att_y],axis=-1)
    sub = x - att_y
    mul = x*att_y

    matching_order_map = {
    "csm":  [cat,sub,mul],
    "cms":  [cat,mul,sub],
    "smc":  [sub,mul,cat],
    "scm":  [sub,cat,mul],
    "mcs":  [mul,cat,sub],
    "msc":  [mul,sub,cat],

    "cs":  [cat,sub],
    "cm":  [cat,mul],
    "sm":  [sub,mul],
    "sc":  [sub,cat],
    "mc":  [mul,cat],
    "ms":  [mul,sub],

    "c":  [cat],
    "s":  [sub],
    "m":  [mul],
    }
    compare_feature = matching_order_map[self.matching_order]
    prev_mem = tf.zeros_like(x)
    #memories=[]
    for i in range(len(compare_feature)):
      with tf.variable_scope("com-fnn-%d"%(i)) as scope:
        linear_com = self.tensordot(inp=compare_feature[i],
                              out_dim= self.config.hidden_units,
                              activation=tf.nn.relu,
                              use_bias=True,
                              w_name="com-fnn")
      prev_mem = update_memory(x=linear_com, prev_mem=prev_mem,epoch=i,use_gated=self.config.use_gated)
      #memories.append(prev_mem)
      memories = prev_mem
    return memories
     
  def get_variable(self,name,shape):
    W = tf.get_variable(name=name, shape=shape,regularizer=l2_regularizer(self.config.l2_strength))
    return W
    
  def frobenius_diff(self,v1,v2):
    vab = tf.matmul(v1,tf.transpose(v2))
    vab = vab*vab
    vab_sum = tf.reduce_sum(vab,1,keep_dims=True)#(batch_sizex1)
    #lab = tf.constant([1,0,0])
    #labelsliced = tf.slice(self.label,[0,0],[self.config.batch_size,1])#(batch_sizex1)
    norm = tf.multiply(vab_sum,tf.cast(self.label,tf.float32))
    norm = tf.reshape(norm,[-1])
    frob_diff = tf.reduce_sum(norm,0)

    return frob_diff

    
