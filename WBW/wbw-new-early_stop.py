# ==============================================================================
# Function:
# Author:
# date:
# ==============================================================================
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
from config import *
from snli_model import *
import snli_reader as reader

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

  def __init__(self, config, data, name=None):
    self.batch_size = batch_size = config.batch_size
    self.x,self.y,self.label, self.x_len,self.y_len,self.data_len = reader.snli_producer(
        data, config,name=name)
    self.epoch_size = self.data_len // self.batch_size
    self.print_data()  

  def print_data(self):
    print (self.x)
    print (self.y)
    print (self.label)
    print (self.x_len)
    print (self.y_len)
    print ("totoal_len",self.data_len)
    print ("self.epoch_size",self.epoch_size)





def run_epoch(session, model, eval_op=None, verbose=False):
  """Runs the model on the given data."""
  start_time = time.time()
  losses = 0.0
  iters = 0
  acc_total=0.0
  fetches = {
      "acc":model.acc,
      "loss": model.loss,
      "global_step":model.global_step,
  }
  if eval_op is not None:
    fetches["eval_op"] = eval_op
  
  start_time = time.time()
  for step in range(model.input.epoch_size):
    feed_dict = {}
    
    vals = session.run(fetches, feed_dict)
    acc = vals["acc"]
    loss = vals["loss"]
    global_step=vals["global_step"]

    losses += loss
    iters= iters+1
    #print('global_step: %s acc: %s' % (global_step,acc))
    #print ("train and test one ITER  time",(time.time()-start_time)/60.0)
    acc_total += acc
    #if verbose and step %10 == 0:
    #  print('global_step: %s train_acc: %s  batch_train_loss: %s' % (global_step,acc,loss))
    acc_average=acc_total/iters
    loss_average = losses/iters
  return acc_average,loss_average,global_step


def get_config():
  if FLAGS.model == "small":
    return SmallConfig()
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
    initializer = tf.random_uniform_initializer(-config.init_scale,
                                                config.init_scale)

    with tf.name_scope("Train"):
      train_input = SNLIInput(config=config, data=Train, name="TrainInput")
      with tf.variable_scope("Model", reuse=None, initializer=initializer):
        m = SNLIModel(is_training=True, config=config, input_=train_input)
      tf.summary.scalar("Training Loss", m.loss)
    
    with tf.name_scope("Valid"):
      valid_input = SNLIInput(config=config, data=Dev, name="ValidInput")
      with tf.variable_scope("Model", reuse=True, initializer=initializer):
        mvalid = SNLIModel(is_training=False,config=config, input_=valid_input)
      tf.summary.scalar("Validation Loss", mvalid.loss)

    with tf.name_scope("Test"):
      test_input = SNLIInput(config=eval_config, data=Test, name="TestInput")
      with tf.variable_scope("Model", reuse=True, initializer=initializer):
        mtest = SNLIModel(is_training=False, config=eval_config,
                         input_=test_input)
   
    sv = tf.train.Supervisor(logdir=FLAGS.save_path)
    with sv.managed_session() as session:
      t0=time.time()
      #sys.exit()
      best_accuracy = config.best_accuracy
      best_val_epoch = config.best_val_epoch

      for i in range(config.MAXITER):
        start_time=time.time()
        train_acc,train_loss,train_global_step = run_epoch(session, m, eval_op=m.optim, verbose=True)
        print("Epoch: %d train_acc: %.3f train_loss %.3f train_global_step:%s" % (i + 1,train_acc,train_loss,train_global_step))
        
        dev_acc,dev_loss,_= run_epoch(session, mvalid)
        print("Epoch: %d dev_acc: %.3f dev_loss %.3f" % (i + 1, dev_acc,dev_loss))

        test_acc,test_loss,_ = run_epoch(session, mtest)
        print("Epoch: %d test_acc: %.3f test_loss %.3f" % (i + 1, test_acc,test_loss))
        
        w_h=run_epoch(session,m)
        print(w_h) 
        # if <= then update 
        if best_accuracy <= dev_acc:
          best_accuracy = dev_acc
          best_val_epoch = i
          if FLAGS.save_path:
            print("train_global_stetp:%s.  Saving %d model to %s." % (train_global_step,i,FLAGS.save_path))
            sv.saver.save(session,"model_saved/model", global_step=train_global_step)
            print (time.strftime("%Y-%m-%d %H:%M:%S",time.localtime()))

        end_time=time.time()
        print("################# all_training time: %s one_epoch time: %s ############### " % ((end_time-t0)//60, (end_time-start_time)//60))
        if i - best_val_epoch > config.early_stopping:
          print ("best_val_epoch:%d  best_accuracy:%s"%(best_val_epoch+1,best_accuracy))
          logging.info("Normal Early stop")
          break        
      
if __name__ == "__main__":
  tf.app.run()
