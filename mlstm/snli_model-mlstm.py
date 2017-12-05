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
import snli_reader as reader

class SNLIModel(object):
  """The SNLI model."""

  def __init__(self, is_training, config, input_):
    self._input = input_
    batch_size = config.batch_size
    self.config = config
    self.is_training = is_training
    self.global_step = tf.Variable(0, trainable=False)

    with tf.device("/cpu:0"):
      embedding_matrix=np.load("glove/snli_glove.npy")
      embedding = tf.Variable(embedding_matrix,trainable=False, name="embedding")
      
      input_xemb = tf.nn.embedding_lookup(embedding, input_.x)
      input_yemb = tf.nn.embedding_lookup(embedding,input_.y)
    
      if is_training and config.keep_prob < 1:
        input_xemb = tf.nn.dropout(input_xemb, config.keep_prob)
        input_yemb = tf.nn.dropout(input_yemb, config.keep_prob)

    with tf.variable_scope("encode_x"):
      self.fwd_lstm_p = self.create_cell()
      self.x_output, self.x_state = tf.nn.dynamic_rnn(cell=self.fwd_lstm_p, inputs=input_xemb,dtype=tf.float32)
      self.x_output =batch_norm(self.x_output,is_training=is_training, updates_collections=None)

    with tf.variable_scope("encode_y"):
      self.fwd_lstm_h = self.create_cell()
      self.y_output, self.y_state = tf.nn.dynamic_rnn(cell=self.fwd_lstm_h, inputs=input_yemb, dtype=tf.float32)
      self.y_output =batch_norm(self.y_output,is_training=is_training, updates_collections=None)

    self.Y = self.x_output  # its length must be x_length
    if is_training and config.keep_prob < 1:
      self.Y = tf.nn.dropout(self.Y,config.keep_prob)  # its length must be x_length
      self.y_output = tf.nn.dropout(self.y_output, config.keep_prob)
    #from batch_major to time_major,fetch last step
    self.y_output= tf.transpose(self.y_output, [1, 0, 2])


    with tf.variable_scope("WbW_attetnion"):
      self.W_Y = tf.get_variable("W_Y", shape=[config.hidden_units, config.hidden_units],regularizer=l2_regularizer(self.config.l2_strength))
      WyY_tmp = tf.matmul(tf.reshape(self.Y, shape=[config.batch_size * config.xmaxlen, config.hidden_units]), self.W_Y,name="WyY")
      self.WyY = tf.reshape(WyY_tmp, shape=[config.batch_size, config.xmaxlen, config.hidden_units])
      r = tf.zeros([self.config.batch_size , self.config.hidden_units])
      
      self.r_all = [] 
      self.m_lstm_cell = self.create_cell()
      m_lstm_state = self.m_lstm_cell.zero_state(self.config.batch_size,dtype=tf.float32)
      for i in range(self.config.ymaxlen):
        if i > 0:
          variable_scope.get_variable_scope().reuse_variables()
        ht = self.y_output[i]
        r,m_lstm_state = self.step_attention(self.m_lstm_cell,m_lstm_state,ht,r) 
        self.r_all.append(r)

      #convet list to 3d tensor
      self.r_all =tf.reshape(self.r_all, shape=[self.config.ymaxlen,self.config.batch_size,-1])

    with tf.variable_scope("matching-lstm"):
        #m_outputs=self.mlstm(self.y_output,self.r_all)

        #self.h_N=self.get_h_n(self.y_output,input_.y_len)
        #self.r_N=self.get_h_n(self.r_all,input_.y_len) 
        self.m_N=self.get_h_n(self.r_all,input_.y_len) 
    
    with tf.variable_scope("pred_layer"):
      # TODO compute h*=tanh(Wp*r+wx*hN)
      #self.W_p = tf.get_variable("W_p", shape=[config.hidden_units, config.hidden_units],regularizer=l2_regularizer(self.config.l2_strength))
      #self.b_p =tf.get_variable("b_p",shape=[config.hidden_units],initializer=tf.constant_initializer())
  
      #self.W_x = tf.get_variable("W_x", shape=[config.hidden_units, config.hidden_units],regularizer=l2_regularizer(self.config.l2_strength))
      #self.b_x = tf.get_variable("b_x",shape=[config.hidden_units],initializer=tf.constant_initializer())

      #self.WprN = tf.matmul(self.r_N, self.W_p, name="WprN") + self.b_p
      #self.WxhN = tf.matmul(self.h_N, self.W_x, name="WxhN") + self.b_x

      #self.hstar = tf.tanh(tf.add(self.WprN, self.WxhN), name="hstar")
      #self.hstar =batch_norm(self.hstar,is_training=is_training, updates_collections=None)
    
      self.W_pred = tf.get_variable("W_pred", shape=[config.hidden_units, 3],regularizer=l2_regularizer(self.config.l2_strength))
      #self.pred = tf.nn.softmax(tf.matmul(self.hstar, self.W_pred), name="pred_layer")
      self.pred = tf.nn.softmax(tf.matmul(self.m_N, self.W_pred), name="pred_layer")

    correct = tf.equal(tf.argmax(self.pred,1),tf.argmax(input_.label,1))
    self.acc = tf.reduce_mean(tf.cast(correct, "float"), name="accuracy")
    
    self.loss_term = -tf.reduce_sum(tf.cast(self._input.label,tf.float32) * tf.log(self.pred),name="loss_term")
    self.reg_term = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES),name="reg_term")
    self.loss = tf.add(self.loss_term,self.reg_term,name="loss")

    if not is_training:
        return 
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars),
                                      config.max_grad_norm)
    optimizer = tf.train.AdamOptimizer(config.learning_rate)
    self.optim = optimizer.apply_gradients(
        zip(grads, tvars),
        global_step=self.global_step)
    _ = tf.summary.scalar("loss", self.loss)

  def step_attention(self,m_cell,state,ht,prev_r):

    self.W_h = tf.get_variable("W_h", shape=[self.config.hidden_units, self.config.hidden_units],regularizer=l2_regularizer(self.config.l2_strength))
    Whht_tmp = tf.matmul(tf.reshape(ht, shape=[self.config.batch_size, self.config.hidden_units]), self.W_h)
    self.Whht = tf.reshape(Whht_tmp, shape=[self.config.batch_size,1, self.config.hidden_units], name="WhhN")

    self.W_r = tf.get_variable("W_r", shape=[self.config.hidden_units, self.config.hidden_units],regularizer=l2_regularizer(self.config.l2_strength))
    Wr_prev_r_tmp = tf.matmul(tf.reshape(prev_r, shape=[self.config.batch_size, self.config.hidden_units]),self.W_r)
    self.Wr_prev_r = tf.reshape(Wr_prev_r_tmp, shape=[self.config.batch_size,1, self.config.hidden_units],name="WhhN")
    
    self.Mt = tf.tanh(self.WyY + self.Whht + self.Wr_prev_r, name="Mt")
    # use attention,TODO compute alpha=softmax(WT*M)
    # need 1 here so that later can do multiplication with h x L
    self.WT = tf.get_variable("WT", shape=[self.config.hidden_units, 1], regularizer=l2_regularizer(self.config.l2_strength))  # h x 1
    WTMt = tf.matmul(tf.reshape(self.Mt, shape=[self.config.batch_size * self.config.xmaxlen, self.config.hidden_units]), self.WT)

    self.alpha = tf.nn.softmax(tf.reshape(WTMt, shape=[self.config.batch_size, 1, self.config.xmaxlen], name="alpha"))  # nb x 1 x Xmax
    self.Y_alpha = tf.reshape(tf.matmul(self.alpha, self.Y),shape=[self.config.batch_size,self.config.hidden_units])  # [nb x 1 x Xmax]*[nb x Xmax x hidden_units] =[nb x 1 x hidden_units]

    # TODO compute r=(Y*alpha) + tanh(Wt*r(t-1))
    #self.W_t = tf.get_variable("W_t", shape=[self.config.hidden_units, self.config.hidden_units],regularizer=l2_regularizer(self.config.l2_strength))
    #self.Wt_prev_r = tf.matmul(prev_r,self.W_t)  # (batch_size,hidden_units)
    #self.rt = self.Y_alpha + tf.tanh(self.Wt_prev_r)
    
    # TODO compute hm_k,use lstm to compute

    mk=tf.concat([self.Y_alpha,ht],axis=1)  #[batch_size,2*hidden_units]
    self.rt,state = m_cell(mk,state)
  
    return self.rt,state

  def mlstm(self,y_output,r_all):
    ''' y_output=[y_seq_len,batch_size,hidden_units]
        r_all =[y_seq_len,batch_size,hidden_units]'''
    m=tf.concat([y_output,r_all],axis=2)
    with tf.variable_scope("mlstm_scope"):
      self.lstm_cell = self.create_cell()
      self.m_outputs, self.m_state = tf.nn.dynamic_rnn(cell=self.lstm_cell, inputs=m,dtype=tf.float32)

    return self.m_outputs
  


  def get_h_n(self,lstm_y_output ,true_y_length):
    ''' lstm_y_output: A Tensor of shape(seq_len, batch_size,hidden_units)(list will work)
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

  @property
  def input(self):
    return self._input

