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

from myutils import *
import snli_reader_mask as reader

class SNLIModel(object):
  """The SNLI model."""

  def __init__(self, is_training, config):

    batch_size = config.batch_size
    self.config = config
    self.is_training = is_training
    self.global_step = tf.Variable(0, trainable=False)
    
    self.x = tf.placeholder(tf.int32, [self.config.batch_size, self.config.xmaxlen])
    self.y = tf.placeholder(tf.int32, [self.config.batch_size, self.config.ymaxlen])

    self.x_mask = tf.placeholder(tf.int32, [self.config.batch_size, self.config.xmaxlen])
    self.y_mask = tf.placeholder(tf.int32, [self.config.batch_size, self.config.ymaxlen])
    self.x_mask = tf.cast(self.x_mask,tf.float32)
    self.y_mask = tf.cast(self.y_mask,tf.float32)

    self.x_len = tf.placeholder(tf.int32, [self.config.batch_size,])
    self.y_len = tf.placeholder(tf.int32, [self.config.batch_size,])
    self.x_len = tf.cast(self.x_len,tf.float32)
    self.y_len = tf.cast(self.y_len,tf.float32)

    self.label = tf.placeholder(tf.int32, [self.config.batch_size,self.config.num_classes])

    with tf.device("/cpu:0"):
      embedding_matrix=np.load("../../data/glove/snli_glove.npy")
      #embedding_matrix=np.load(self.config.glove_dir)
      embedding = tf.Variable(embedding_matrix,trainable=False, name="embedding")
      
      input_xemb = tf.nn.embedding_lookup(embedding, self.x)
      input_yemb = tf.nn.embedding_lookup(embedding,self.y)
    
      if is_training and config.keep_prob < 1:
        input_xemb = tf.nn.dropout(input_xemb, config.keep_prob)
        input_yemb = tf.nn.dropout(input_yemb, config.keep_prob)

    with tf.variable_scope("encode_x"):
      self.x_output=self.two_layer_tensordot(input_xemb,self.config.hidden_units,scope="x_fnn")
      self.x_output=self.x_output*self.x_mask[:,:,None]

    with tf.variable_scope("encode_y"):
      self.y_output=self.two_layer_tensordot(input_yemb,self.config.hidden_units,scope="y_fnn")
      self.y_output=self.y_output*self.y_mask[:,:,None]
    if is_training and config.keep_prob < 1:
      self.x_output = tf.nn.dropout(self.x_output,config.keep_prob)  # its length must be x_length
      self.y_output = tf.nn.dropout(self.y_output, config.keep_prob)
    
    with tf.variable_scope("dot-product-atten"):
      #weightd_y:(b,x_len,h),weighted_x:(b,y_len,h)
      self.weighted_y, self.weighted_x =self.dot_product_attention(x_sen=self.x_output,y_sen=self.y_output,x_len = self.config.xmaxlen,y_len=self.config.ymaxlen)

    with tf.variable_scope("collect-info"):

      co_xy = tf.concat([self.x_output,self.weighted_y],axis=-1) #(b,x_len,2*h)
      co_yx = tf.concat([self.y_output,self.weighted_x],axis=-1) #(b,y_len,2*h)
      v_co_xy=self.two_layer_tensordot(co_xy,self.config.hidden_units,scope="compare_fnn_1")
      v_co_xy=v_co_xy*self.x_mask[:,:,None]
      v_co_yx=self.two_layer_tensordot(co_yx,self.config.hidden_units,scope="compare_fnn_2")
      v_co_yx=v_co_yx*self.y_mask[:,:,None]
      if is_training and config.keep_prob < 1:
        v_co_xy = tf.nn.dropout(v_co_xy,config.keep_prob)  #(b,x_len,h)
        v_co_yx = tf.nn.dropout(v_co_yx,config.keep_prob)  #(b,x_len,h)


    with tf.variable_scope("pooling"):
      #irrelevant with seq_len,keep the final dims
      v1=tf.reduce_sum(v_co_xy,axis=1)#(b,h)
      v2=tf.reduce_sum(v_co_yx,axis=1)#(b,h)

      self.v = tf.concat([v1,v2],axis=-1) #(b,2*h)

    with tf.variable_scope("pred-layer"):
      fnn1 = self.fnn(input=self.v,
                      out_dim=self.config.hidden_units,
                      activation=tf.nn.tanh,
                      use_bias=True,
                      w_name="fnn-pred-W")
  
      if is_training and config.keep_prob < 1:
        fnn1 = tf.nn.dropout(fnn1, config.keep_prob)

      W_pred = tf.get_variable("W_pred", shape=[self.config.hidden_units, 3],regularizer=l2_regularizer(self.config.l2_strength))
      self.pred = tf.nn.softmax(tf.matmul(fnn1, W_pred), name="pred")

      correct = tf.equal(tf.argmax(self.pred,1),tf.argmax(self.label,1))
      self.acc = tf.reduce_mean(tf.cast(correct, "float"), name="accuracy")
    
      self.loss_term = -tf.reduce_sum(tf.cast(self.label,tf.float32) * tf.log(self.pred),name="loss_term")
      self.reg_term = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES),name="reg_term")
      self.loss = tf.add(self.loss_term,self.reg_term,name="loss")

    if not is_training:
        return
    with tf.variable_scope("bp_layer"):
      tvars = tf.trainable_variables()
      grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars),
                                      config.max_grad_norm)
      optimizer = tf.train.AdamOptimizer(config.learning_rate)
      self.optim = optimizer.apply_gradients(
          zip(grads, tvars),
          global_step=self.global_step)
      #_ = tf.summary.scalar("loss", self.loss)

  def dot_product_attention(self,x_sen,y_sen,x_len,y_len):
    '''
    function: use the dot-production of left_sen and right_sen to compute the attention weight matrix
    :param left_sen: a list of 2D tensor (x_len,hidden_units)
    :param right_sen: a list of 2D tensor (y_len,hidden_units)
    :return: (1) weighted_y: the weightd sum of y_sen, a 3D tensor with shape (b,x_len,2*h)
             (2)weghted_x:  the weighted sum of x_sen, a 3D tensor with shape (b,y_len,2*h)
    '''
    
    weight_matrix =tf.matmul(x_sen, tf.transpose(y_sen,perm=[0,2,1])) #(b,x_len,h) x (b,h,y_len)->(b,x_len,y_len)

    weight_matrix_y =tf.exp(weight_matrix - tf.reduce_max(weight_matrix,axis=2,keep_dims=True))  #(b,x_len,y_len)
    weight_matrix_x =tf.exp(tf.transpose((weight_matrix - tf.reduce_max(weight_matrix,axis=1,keep_dims=True)),perm=[0,2,1]))  #(b,y_len,x_len)

    weight_matrix_y=weight_matrix_y*self.y_mask[:,None,:]#(b,x_len,y_len)*(b,1,y_len)
    weight_matrix_x=weight_matrix_x*self.x_mask[:,None,:]#(b,y_len,x_len)*(b,1,x_len)
    
    alpha=weight_matrix_y/(tf.reduce_sum(weight_matrix_y,2,keep_dims=True)+1e-8)#(b,x_len,y_len)
    beta=weight_matrix_x/(tf.reduce_sum(weight_matrix_x,2,keep_dims=True)+1e-8)#(b,y_len,x_len)

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
      
      W = tf.get_variable(w_name,shape=[in_dim,out_dim])
      out = tf.tensordot(inp,W,axes=1)

      if use_bias == True:
        b_name = w_name + '-b'
        b = tf.get_variable(b_name, shape=[out_dim])
        out = out + b

      if activation is not None:
        out = activation(out)
      out.set_shape([batch_size,seq_len,out_dim])
      return out
  def two_layer_tensordot(self,inp,out_dim,scope):
    with tf.variable_scope(scope):
      output1=self.tensordot(inp,out_dim,activation=tf.nn.relu,use_bias=True,w_name="first_layer")
      output2=self.tensordot(output1,out_dim,activation=tf.nn.relu,use_bias=True,w_name="second_layer")
      return output2
    
  def create_cell(self):
    
    def lstm_cell():
      # With the latest TensorFlow source code (as of Mar 27, 2017),
      # the BasicLSTMCell will need a reuse parameter which is unfortunately not
      # defined in TensorFlow 1.0. To maintain backwards compatibility, we add
      # an argument check here:
      if 'reuse' in inspect.getargspec(
          tf.contrib.rnn.BasicLSTMCell.__init__).args:
        return tf.contrib.rnn.BasicLSTMCell(
            self.config.hidden_units, forget_bias=0.0, state_is_tuple=True,
            reuse=tf.get_variable_scope().reuse)
        print("reuse lstm cell")
      else:
        print ("not reuse lstm cell")
        return tf.contrib.rnn.BasicLSTMCell(
            self.config.hidden_units, forget_bias=0.0, state_is_tuple=True)
    attn_cell = lstm_cell

    if self.is_training and self.config.keep_prob < 1:
      def attn_cell():
        return tf.contrib.rnn.DropoutWrapper(
            lstm_cell(), output_keep_prob=self.config.keep_prob)
    cell = tf.contrib.rnn.MultiRNNCell(
        [attn_cell() for _ in range(self.config.num_layers)], state_is_tuple=True)
    return cell

  def my_dynamic_rnn(self,inp,mask):
    with tf.variable_scope("my_rnn"):
      self.my_cell= self.create_cell()
      self.output, self.state = tf.nn.dynamic_rnn(cell=self.my_cell, inputs=inp,dtype=tf.float32) #(b,s,h)
      self.output=self.output*mask[:,:,None]
      self.output =batch_norm(self.output,is_training=is_training, updates_collections=None)
    return self.output,self.state

  def my_bidirectional_dynamic_rnn(self,inp,mask):
    with tf.variable_scope("my_bi_rnn"):
      mask = tf.cast(mask, tf.float32)     

      self.my_cell_fw=self.create_cell()
      self.my_cell_bw=self.create_cell()
      self.outputs,self.states= tf.nn.bidirectional_dynamic_rnn(cell_fw=self.my_cell_fw,cell_bw=self.my_cell_bw,inputs=inp, dtype=tf.float32)
      self.output_fw,self.output_bw=self.outputs
      self.state_fw,self.state_bw=self.states
      self.output_fw=self.output_fw*mask[:,:,None]    
      self.output_bw=self.output_bw*mask[:,:,None]

      self.output_fw =batch_norm(self.output_fw,is_training=self.is_training, updates_collections=None)
      self.output_bw =batch_norm(self.output_bw,is_training=self.is_training, updates_collections=None)
      return self.output_fw,self.output_bw,self.state_fw,self.state_bw

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

  def direct2pred(self,arg1):
    '''
    args:arg1: a 2D tensor of shape (batch_size,hidden_units)
    function: softmax(W*arg1)
    return: pred
    '''
    with tf.variable_scope("direct2predict_layer"):
      h_predict= arg1
      W_pred = tf.get_variable("concat_W_pred", shape=[self.config.hidden_units, 3],regularizer=l2_regularizer(self.config.l2_strength))
      pred = tf.nn.softmax(tf.matmul(h_predict, W_pred), name="pred")
      return pred

  def concat2pred(self,arg1,arg2):
    '''
    args:arg1/arg2: a 2D tensor of shape (batch_size,hidden_units)
    function: softmax(W*concat(arg1,arg2))
    return: pred
    '''
    with tf.variable_scope("concat2predict_layer"):
      h_predict= tf.concat([arg1,arg2],axis=1,name="h_predict")
      W_pred = tf.get_variable("concat_W_pred", shape=[2*self.config.hidden_units, 3],regularizer=l2_regularizer(self.config.l2_strength))
      pred = tf.nn.softmax(tf.matmul(h_predict, W_pred), name="pred")
      return pred

  def weighted2pred(self,arg1,arg2,bias=True,activation=None):
    ''' TODO compute h*=tanh(W1*arg1+W2*arg2)
             softmax(W x (h*))
    '''
    with tf.variable_scope("weighted2predict_layer"):
      weighted_arg1= tf.layers.dense(inputs=arg1,
                      units=self.config.hidden_units,
                      activation=None,
                      use_bias=False,
                      kernel_regularizer=l2_regularizer(self.config.l2_strength),
                      name="weight_arg1")
      weighted_arg2= tf.layers.dense(inputs=arg2,
                      units=self.config.hidden_units,
                      activation=None,
                      use_bias=False,
                      kernel_regularizer=l2_regularizer(self.config.l2_strength),
                      name="weight_arg2")
      if acivation is not None:
        hstar = activation(tf.add(weight_arg1,weight_arg2,name="hstar"))
      else:
        hstar = tf.add(weight_arg1,weight_arg2,name="hstar")

      h_predict= tf.layers.dense(inputs=hstar,
                      units=self.config.num_classes,
                      activation=None,
                      use_bias=True,
                      kernel_regularizer=l2_regularizer(self.config.l2_strength),
                      name="h_predict")
      pred = tf.nn.softmax(h_predict,name="pred")
      return pred
