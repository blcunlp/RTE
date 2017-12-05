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

from ops_cudnn_bilstm import *

from myutils import *
import snli_reader_mask as reader

class SNLIModel(object):
  """The SNLI model."""

  def __init__(self, is_training, config):

    batch_size = config.batch_size
    self.config = config
    self.is_training = is_training
    self.global_step = tf.Variable(0, trainable=False)

    self.learning_rate = tf.Variable(self.config.learning_rate, trainable=False)
    self._new_lr = tf.placeholder(tf.float32, shape=[], name="new_learning_rate")
    self._lr_update = tf.assign(self.learning_rate, self._new_lr)

    
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
      embedding_matrix=np.load("../data/glove/snli_glove.npy")
      #embedding_matrix=np.load(self.config.glove_dir)
      embedding = tf.Variable(embedding_matrix,trainable=False, name="embedding")
      
      input_xemb = tf.nn.embedding_lookup(embedding, self.x)
      input_yemb = tf.nn.embedding_lookup(embedding,self.y)
    
      if is_training and config.keep_prob < 1:
        input_xemb = tf.nn.dropout(input_xemb, config.keep_prob)
        input_yemb = tf.nn.dropout(input_yemb, config.keep_prob)

    with tf.variable_scope("encode_x") as scope:
      self.x_output,_,_ = cudnn_lstm(inputs=input_xemb,num_layers=1,hidden_size=self.config.hidden_units,is_training=self.is_training)    

      scope.reuse_variables()
      #with tf.variable_scope("encode_y"):
      self.y_output,_,_ = cudnn_lstm(inputs=input_yemb,num_layers=1,hidden_size=self.config.hidden_units,is_training=self.is_training)    

      #self.y_output=tf.concat(y_bilstm,-1)
      if is_training and config.keep_prob < 1:
        self.x_output = tf.nn.dropout(self.x_output, config.keep_prob)
        self.y_output = tf.nn.dropout(self.y_output, config.keep_prob)



    print ("x_output",self.x_output)
    print ("y_output",self.y_output)
    
    with tf.variable_scope("dot-product-atten") :
      #weightd_y:(b,x_len,2*h),weighted_x:(b,y_len,2*h)
      self.weighted_y, self.weighted_x =self.dot_product_attention(x_sen=self.x_output,y_sen=self.y_output,x_len = self.config.xmaxlen,y_len=self.config.ymaxlen)

    with tf.variable_scope("collect-info"):
      diff_xy = tf.subtract(self.x_output,self.weighted_y) #Returns x - y element-wise.
      diff_yx = tf.subtract(self.y_output,self.weighted_x)

      mul_xy = tf.multiply(self.x_output,self.weighted_y)
      mul_yx = tf.multiply(self.y_output, self.weighted_x)
  
      m_xy = tf.concat([self.x_output,self.weighted_y,diff_xy,mul_xy],axis=2) #(b,x_len,8*h)
      m_yx = tf.concat ([self.y_output,self.weighted_x,diff_yx,mul_yx],axis=2) #(b,y_len,8*h)
      with tf.variable_scope("aggregate-fnn") as scope:
        m_xy = self.tensordot(inp=m_xy,
                            out_dim= self.config.hidden_units,
                            activation=tf.nn.relu,
                            use_bias=True,
                            w_name="fnn-mxy_W")
        scope.reuse_variables()
        m_yx = self.tensordot(inp=m_yx,
                            out_dim= self.config.hidden_units,
                            activation=tf.nn.relu,
                            use_bias=True,
                            w_name="fnn-mxy_W")

      if is_training and config.keep_prob < 1:
        m_xy = tf.nn.dropout(m_xy,config.keep_prob)  
        m_yx = tf.nn.dropout(m_yx,config.keep_prob)  

    with tf.variable_scope("composition"):

      with tf.variable_scope("encode_mxy") as scope:
        mxy_output,_,_ = cudnn_lstm(inputs=m_xy,num_layers=1,hidden_size=self.config.hidden_units,is_training=self.is_training)    

        scope.reuse_variables()

        #with tf.variable_scope("encode_myx"):
 
        myx_output,_,_ = cudnn_lstm(inputs=m_yx,num_layers=1,hidden_size=self.config.hidden_units,is_training=self.is_training)    


    with tf.variable_scope("pooling"):
      #irrelevant with seq_len,keep the final dims
      v_xymax = tf.reduce_max(mxy_output,axis=1)  #(b,2h)    
      v_xy_sum = tf.reduce_sum(mxy_output, 1)  #(b,x_len.2*h) ->(b,2*h)
      v_xyave = tf.div(v_xy_sum, tf.expand_dims(self.x_len, -1)) #div true length

      v_yxmax = tf.reduce_max(myx_output,axis=1)  #(b,2h)
      v_yx_sum = tf.reduce_sum(myx_output, 1)   ##(b,y_len.2*h) ->(b,2*h)
      v_yxave = tf.div(v_yx_sum, tf.expand_dims(self.y_len,  -1)) #div true length
      #v_xyave = tf.reduce_mean(mxy_output,axis=1) #(b,2h)
      #v_yxave = tf.reduce_mean(myx_output,axis=1) #(b,2h)

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
      for var in tvars:
        print (var.name,var.shape)
      grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars),
                                      config.max_grad_norm)
      optimizer = tf.train.AdamOptimizer(self.learning_rate)
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
    
    alpha=weight_matrix_y/(tf.reduce_sum(weight_matrix_y,1,keep_dims=True)+1e-8)#(b,x_len,y_len)
    beta=weight_matrix_x/(tf.reduce_sum(weight_matrix_x,1,keep_dims=True)+1e-8)#(b,y_len,x_len)

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

  def batch_product_3D(self,inp1,inp2):
    '''
    function: compute 3D mtrix multiplication, (b,m,n)*(b,n,d)=>(b,m,d)
              inp1 act as the weight matrix of inp2
    :param inp1: a 3D tensor of (b,x_len,y_len)
    :param inp2: a 3D tensor of (b,y_len,dim)
    :return: the weighted sum of inp1*inpu2, a 3D tensor with shape (b,x_len,dim)
    '''
    def singel_instance(x):
      '''
      :param x: the weight and input of one time step
      :return: the matrix multiplution(attention) of one time step
      '''
      w_qt = x[0]  # w_q:(x_len,y_len)
      qt = x[1]    # q:(y_len,dim)

      weighted_qt= tf.matmul(w_qt,qt)  #(x_len,dim)
      return weighted_qt

    elems = (inp1, inp2)
    return tf.map_fn(singel_instance, elems, dtype=tf.float32)  # [b, x_len, dim]

  def matching_lstm(self,encoder_outputs,decoder_inputs,en_sequence_len,de_sequence_len):
    '''
    :param encoder_outputs: a  list of 2D tensor with shape of (batch_size, sey_len,hidden_units)
    :param decoder_inputs: a  list of 2D tensor with shape of (batch_size, sey_len,hidden_units)
    :param en_sequence_len: the max sequence length of encoder_outputs
    :param de_sequence_len: the max sequence length of decoder_outputs
    :return: the result of matching decoder and encoder context
    '''
    with tf.variable_scope("matching_encoder_decoder"):
      decoder_inputs = tf.transpose(decoder_inputs, [1, 0, 2])  # (b,s,h) ->(s,b,h)
      W_Y = tf.get_variable("W_Y", shape=[self.config.hidden_units, self.config.hidden_units],
                                 regularizer=l2_regularizer(self.config.l2_strength))
      WyY_tmp = tf.matmul(tf.reshape(encoder_outputs, shape=[self.config.batch_size * en_sequence_len, self.config.hidden_units]), W_Y,
                          name="WyY")
      weighted_encoder_outputs= tf.reshape(WyY_tmp, shape=[self.config.batch_size, en_sequence_len, self.config.hidden_units])


      def step_attention(cell, state, inp, prev_output):

        W_h = tf.get_variable("W_h", shape=[self.config.hidden_units, self.config.hidden_units],
                              regularizer=l2_regularizer(self.config.l2_strength))
        Whht_tmp = tf.matmul(tf.reshape(inp, shape=[self.config.batch_size, self.config.hidden_units]), W_h)
        Whht = tf.reshape(Whht_tmp, shape=[self.config.batch_size, 1, self.config.hidden_units], name="WhhN")

        W_r = tf.get_variable("W_r", shape=[self.config.hidden_units, self.config.hidden_units],
                              regularizer=l2_regularizer(self.config.l2_strength))
        Wr_prev_output_tmp = tf.matmul(tf.reshape(prev_output, shape=[self.config.batch_size, self.config.hidden_units]), W_r)
        Wr_prev_output = tf.reshape(Wr_prev_output_tmp, shape=[self.config.batch_size, 1, self.config.hidden_units], name="WhhN")

        Mt = tf.tanh(weighted_encoder_outputs + Whht + Wr_prev_output, name="Mt")
        # use attention,TODO compute alpha=softmax(WT*M)
        # need 1 here so that later can do multiplication with h x L
        WT = tf.get_variable("WT", shape=[self.config.hidden_units, 1],
                             regularizer=l2_regularizer(self.config.l2_strength))  # h x 1
        WTMt = tf.matmul(tf.reshape(Mt, shape=[self.config.batch_size * en_sequence_len, self.config.hidden_units]),
                         WT)

        alpha = tf.nn.softmax(
          tf.reshape(WTMt, shape=[self.config.batch_size, 1, en_sequence_len], name="alpha"))  # nb x 1 x Xmax
        ct = tf.reshape(tf.matmul(alpha, encoder_outputs), shape=[self.config.batch_size,
                                                         self.config.hidden_units])  # [nb x 1 x Xmax]*[nb x Xmax x hidden_units] =[nb x 1 x hidden_units]

        # TODO compute hm_k,use lstm to compute
        mk = tf.concat([ct, inp], axis=1)  # [batch_size,2*hidden_units]
        cell_output, state = cell(mk, state)

        return cell_output, state

      match_h = []
      m_lstm_cell = self.create_cell()
      m_lstm_state = m_lstm_cell.zero_state(self.config.batch_size, dtype=tf.float32)
      cell_output = tf.zeros([self.config.batch_size, self.config.hidden_units])
      for i in range(de_sequence_len):
        if i > 0:
          variable_scope.get_variable_scope().reuse_variables()
        inp = decoder_inputs[i]
        cell_output, m_lstm_state = step_attention(m_lstm_cell, m_lstm_state, inp, cell_output)
        match_h.append(cell_output)
      return  match_h

  def self_matching_lstm(self,encoder_outputs,decoder_inputs,en_sequence_len,de_sequence_len):
    '''
    :param encoder_outputs: a  list of 2D tensor with shape of ( batch_size,hidden_units)
    :param decoder_inputs: a  list of 2D tensor with shape of (batch_size,hidden_units)
    :param en_sequence_len: the max sequence length of encoder_outputs
    :param de_sequence_len: the max sequence length of decoder_outputs
    :return: the result of matching decoder and encoder context
    '''
    with tf.variable_scope("matching_encoder_decoder"):
      encoder_outputs_transpose = tf.transpose(encoder_outputs, [1, 0, 2])  # (s,b,h) ->(b,s,h)
      W_Y = tf.get_variable("W_Y", shape=[self.config.hidden_units, self.config.hidden_units],
                                 regularizer=l2_regularizer(self.config.l2_strength))
      WyY_tmp = tf.matmul(tf.reshape(encoder_outputs, shape=[self.config.batch_size * en_sequence_len, self.config.hidden_units]), W_Y,
                          name="WyY")
      weighted_encoder_outputs= tf.reshape(WyY_tmp, shape=[self.config.batch_size, en_sequence_len, self.config.hidden_units])


      def self_step_attention(cell, state, inp):
        '''
        :param cell:
        :param state:
        :param inp: current input
        :return:
        '''
        W_h = tf.get_variable("W_h", shape=[self.config.hidden_units, self.config.hidden_units],
                              regularizer=l2_regularizer(self.config.l2_strength))
        Whht_tmp = tf.matmul(tf.reshape(inp, shape=[self.config.batch_size, self.config.hidden_units]), W_h)
        Whht = tf.reshape(Whht_tmp, shape=[self.config.batch_size, 1, self.config.hidden_units], name="WhhN")

        W_r = tf.get_variable("W_r", shape=[self.config.hidden_units, self.config.hidden_units],
                              regularizer=l2_regularizer(self.config.l2_strength))
        Mt = tf.tanh(weighted_encoder_outputs + Whht, name="Mt") #(b,e_s,h)
        # use attention,TODO compute alpha=softmax(WT*M)
        # need 1 here so that later can do multiplication with h x L
        WT = tf.get_variable("WT", shape=[self.config.hidden_units, 1],
                             regularizer=l2_regularizer(self.config.l2_strength))  # h x 1
        WTMt = tf.matmul(tf.reshape(Mt, shape=[self.config.batch_size * en_sequence_len, self.config.hidden_units]),
                         WT) #(b*e_s,h)*(h,1)=(b*e_s,1)

        alpha = tf.nn.softmax(
          tf.reshape(WTMt, shape=[self.config.batch_size, 1, en_sequence_len], name="alpha"))  #b x 1 x e_s
        ct = tf.reshape(tf.matmul(alpha, encoder_outputs_transpose), shape=[self.config.batch_size,
                                                         self.config.hidden_units])
        # ct: [b x 1 x e_s]*[b x e_s x h] =[b x 1 x h] ->reshape -->[b,h]

        # TODO compute hm_k,use lstm to compute
        mk = tf.concat([ct, inp], axis=1)  # [batch_size,2*hidden_units]
        cell_output, state = cell(mk, state)

        return cell_output, state

      match_h = []
      m_lstm_cell = self.create_cell()
      m_lstm_state = m_lstm_cell.zero_state(self.config.batch_size, dtype=tf.float32)
      cell_output = tf.zeros([self.config.batch_size, self.config.hidden_units])
      for i in range(de_sequence_len):
        if i > 0:
          variable_scope.get_variable_scope().reuse_variables()
        inp = decoder_inputs[i]
        cell_output, m_lstm_state = self_step_attention(m_lstm_cell, m_lstm_state, inp)
        match_h.append(cell_output)
      return  match_h

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



  def assign_lr(self, session, lr_value):
    session.run(self._lr_update, feed_dict={self._new_lr: lr_value})
  

