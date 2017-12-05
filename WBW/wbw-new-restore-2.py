
# Function:
# Author:
# date:
# ==============================================================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import time
import inspect
import numpy as np
import re
import tensorflow as tf
from tensorflow.contrib.rnn.python.ops import core_rnn_cell
from tensorflow.contrib.layers import batch_norm,l2_regularizer
from tensorflow.python.ops import variable_scope
from myutils import *
import snli_reader as reader
from snli_model_fnn import SNLIModel
#from snli_model import SNLIModel
from config import SmallConfig
from config import TestConfig

flags = tf.flags
logging = tf.logging

flags.DEFINE_string(
    "model", "small",
    "A type of model. Possible options are: small, medium, large.")
flags.DEFINE_string("data_path", "",
                    "Where the training/test data is stored.")
flags.DEFINE_string("save_path","model_saved",
                    "Model output directory.")
flags.DEFINE_bool("use_fp16", False,
                  "Train using 16-bit floats instead of 32bit floats")

FLAGS = flags.FLAGS

def data_type():
  return tf.float16 if FLAGS.use_fp16 else tf.float32

class SNLIInput(object):
  """The input data."""

  def __init__(self, config, data,vocab, name=None):
    self.batch_size = batch_size = config.batch_size
    self.x,self.y,self.label, self.x_len,self.y_len,self.data_len = reader.snli_producer(
        data, config,name=name)
    self.epoch_size = self.data_len // self.batch_size
    self.vocab =vocab
    #self.print_data()

  def print_data(self):
    print (self.x)
    print (self.y)
    print (self.label)
    print (self.x_len)
    print (self.y_len)
    print ("totoal_len",self.data_len)
    print ("self.epoch_size",self.epoch_size)



def post_process(x,y,true_label,pred_label,vocab):
    
  true_label =np.argmax(true_label,1)[0]
  pred_label =np.argmax(pred_label,1)[0]
    
  x_txt =str(id_batch2txt_batch(x,vocab)) 
  y_txt =str(id_batch2txt_batch(y,vocab))
  
  x_txt=x_txt.replace("['","")
  x_txt=x_txt.replace("']","")

  y_txt=y_txt.replace("['","")
  y_txt=y_txt.replace("']","")

  print_out = 'pred_label:{}	true_label:{}	premise:{}	hypothesis:{}\n'.format(pred_label, true_label, x_txt,y_txt)
  if true_label == pred_label:
    write_file("pred_correct.txt",print_out)
  else:
    write_file("pred_wrong.txt",print_out)
  write_file("pred_all.txt",print_out)
  #print (step,print_out) 

  
def write_file(filename,data):
    with open(filename, "a+") as f:
      f.write(data)

def run_epoch(session, model, vocab,eval_op=None, verbose=False):
  start_time = time.time()
  losses = 0.0
  iters = 0
  acc_total=0.0
  fetches = {
      "acc":model.acc,
      "loss": model.loss,
      "global_step":model.global_step,
      "pred_label":model.pred,
      "true_label":model._input.label,
      "x":model._input.x,
      "y":model._input.y,
  }
  if eval_op is not None:
    fetches["eval_op"] = eval_op
  
  start_time = time.time()
  for step in range(model._input.epoch_size):
    feed_dict = {}
    
    vals = session.run(fetches, feed_dict)
    acc = vals["acc"]
    loss = vals["loss"]
    global_step=vals["global_step"]
    pred_label = vals["pred_label"]
    true_label =vals["true_label"]
    x= vals["x"]
    y = vals["y"]

    post_process(x,y,true_label,pred_label,vocab)
    #true_label =np.argmax(true_label,1)
    #pred_label =np.argmax(pred_label,1)
    #x_txt = id_batch2txt_batch(x,vocab) 
    #y_txt = id_batch2txt_batch(y,vocab)
    #print_out = 'pred_label:{}	true_label:{}	premise:{}	hyposis:{}'.format(pred_label, true_label, x_txt,y_txt)
    if step%100 ==0:
      print (step) 
    losses += loss
    iters= iters+1
    acc_total += acc
    if verbose and step %10 == 0:
      print('global_step: %s train_acc: %s  batch_train_loss: %s' % (global_step,acc,loss))
    acc_average=acc_total/iters
    loss_average = losses/iters
 
  return acc_average,loss_average
"""

def run_epoch(session, vocab,eval_op=None, verbose=False):
  start_time = time.time()
  losses = 0.0
  iters = 0
  acc_total=0.0
  fetches = {
      "acc":acc,
      "loss":loss,
      "global_step":global_step,
      "pred_label":pred,
      "true_label":_input.label,
      "x":_input.x,
      "y":_input.y,
  }
  if eval_op is not None:
    fetches["eval_op"] = eval_op
  
  start_time = time.time()
  for step in range(_input.epoch_size):
    feed_dict = {}
    
    vals = session.run(fetches, feed_dict)
    acc = vals["acc"]
    loss = vals["loss"]
    global_step=vals["global_step"]
    pred_label = vals["pred_label"]
    true_label =vals["true_label"]
    x= vals["x"]
    y = vals["y"]

    post_process(x,y,true_label,pred_label,vocab)
    #true_label =np.argmax(true_label,1)
    #pred_label =np.argmax(pred_label,1)
    #x_txt = id_batch2txt_batch(x,vocab) 
    #y_txt = id_batch2txt_batch(y,vocab)
    #print_out = 'pred_label:{}	true_label:{}	premise:{}	hyposis:{}'.format(pred_label, true_label, x_txt,y_txt)
    if step%100 ==0:
      print (step) 
    losses += loss
    iters= iters+1
    acc_total += acc
    if verbose and step %10 == 0:
      print('global_step: %s train_acc: %s  batch_train_loss: %s' % (global_step,acc,loss))
    acc_average=acc_total/iters
    loss_average = losses/iters
  return acc_average,loss_average
"""

def get_config():
  if FLAGS.model == "small":
    return SmallConfig()
    # return TestConfig()
  elif FLAGS.model == "test":
    return TestConfig()
  else:
    raise ValueError("Invalid model: %s", FLAGS.model)

def main(_):
  config = get_config()
  eval_config=get_config()
  eval_config.batch_size=1


  Train,Dev,Test,vocab = reader.file2seqid(config)

  with tf.Graph().as_default():
    initializer = tf.random_uniform_initializer(-config.init_scale,                                            config.init_scale)
    
    with tf.name_scope("Train"):
      train_input = SNLIInput(config=config, data=Train,vocab=vocab,name="TrainInput")
      with tf.variable_scope("Model", reuse=None, initializer=initializer):
        m = SNLIModel(is_training=True, config=config, input_=train_input)
      tf.summary.scalar("Training Loss", m.loss)
    
    with tf.name_scope("Valid"):
      valid_input = SNLIInput(config=config, data=Dev, vocab=vocab,name="ValidInput")
      with tf.variable_scope("Model", reuse=True, initializer=initializer):
        mvalid = SNLIModel(is_training=False,config=config, input_=valid_input)
      tf.summary.scalar("Validation Loss", mvalid.loss)    
    with tf.name_scope("Test"):
      test_input = SNLIInput(config=eval_config, data=Test, vocab=vocab,name="TestInput")
      with tf.variable_scope("Model", reuse=True, initializer=initializer):
        mtest = SNLIModel(is_training=False, config=eval_config,
                         input_=test_input)

    print ("loading model")  
    saver = tf.train.Saver()
 
    def load_pretrain(sess):
      return saver.restore(sess, "model_saved/model.ckpt")

    if os.path.isfile("model_saved/checkpoint") == False:
      print ("pre_trained model is None")
      load_pretrain=None
    else:
     print ("pre_trained model is in checkpoint")
  
    sv = tf.train.Supervisor(logdir=FLAGS.save_path,init_fn = load_pretrain) #logdir用来保存checkpoint和summary

    #sv = tf.train.Supervisor(logdir=FLAGS.save_path)
    with sv.managed_session() as session:
      t0=time.time()
      
      start_time=time.time()
      print ("test run epoch")
      test_acc,test_loss = run_epoch(session, mtest,vocab)
      print("test_acc: %.3f test_loss %.3f" % (test_acc,test_loss))    
        
      end_time=time.time()
      print("training time: %s one_epoch time: %s " % ((end_time-t0)//60, (end_time-start_time)//60))
      #print (time.strftime("%Y-%m-%d %H:%M:%S",time.localtime())) == "__main__":
if __name__ == "__main__":  
  tf.app.run()
