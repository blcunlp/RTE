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
import snli_reader_mask as reader
from snli_model_mask_lr import SNLIModel
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

def fill_placeholder(data, model,config):
  batch_x,batch_y,batch_label,batch_x_mask,batch_y_mask, batch_x_len,batch_y_len= data.next_batch(config.batch_size)
  feed_dict = {model.x:batch_x , 
                model.y:batch_y,
                model.label:batch_label,
                model.x_mask:batch_x_mask,
                model.y_mask:batch_y_mask, 
                model.x_len :batch_x_len,
                model.y_len :batch_y_len,
                }

  return feed_dict

def run_epoch(session, data,model,config, eval_op=None, verbose=False):
  """Runs the model on the given data."""
  start_time = time.time()
  losses = 0.0
  iters = 0
  acc_total=0.0
  fetches = {
      "acc":model.acc,
      "loss": model.loss,
      "global_step":model.global_step,
      "learning_rate": model.learning_rate,
  }
  if eval_op is not None:
    fetches["eval_op"] = eval_op
  
  start_time = time.time()
  epoch_size = data.get_epoch_size(config.batch_size)
  for step in range(epoch_size):
    feed_dict = fill_placeholder(data,model,config)
    
    vals = session.run(fetches, feed_dict)
    acc = vals["acc"]
    loss = vals["loss"]
    global_step=vals["global_step"]

    learning_rate = vals["learning_rate"]

    losses += loss
    iters= iters+1
    #print('global_step: %s acc: %s' % (global_step,acc))
    #print ("train and test one ITER  time",(time.time()-start_time)/60.0)
    acc_total += acc
    #if verbose and step %10 == 0:
    #  print('global_step: %s train_acc: %s  batch_train_loss: %s' % (global_step,acc,loss))
    acc_average=acc_total/iters
    loss_average = losses/iters
  return acc_average,loss_average,global_step,learning_rate


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
      with tf.variable_scope("Model", reuse=None, initializer=initializer):
        m = SNLIModel(is_training=True, config=config)
      #tf.summary.scalar("Training Loss", m.loss)
    
    with tf.name_scope("Valid"):
      with tf.variable_scope("Model", reuse=True, initializer=initializer):
        mvalid = SNLIModel(is_training=False,config=eval_config)
      #tf.summary.scalar("Validation Loss", mvalid.loss)

    with tf.name_scope("Test"):
      with tf.variable_scope("Model", reuse=True, initializer=initializer):
        mtest = SNLIModel(is_training=False, config=eval_config)
    
    sv = tf.train.Supervisor(logdir=FLAGS.save_path)
    with sv.managed_session() as session:
      t0=time.time()
      #sys.exit()
      best_accuracy = config.best_accuracy
      best_val_epoch = config.best_val_epoch
      last_change_epoch = 0

      best_test_acc=0
      best_test_epoch=0

      for i in range(config.MAXITER):
        start_time=time.time()
        train_acc,train_loss,train_global_step,learning_rate = run_epoch(session,data=Train, model=m,config=config, eval_op=m.optim, verbose=True)
        print("Epoch: %d train_acc: %.3f train_loss %.3f train_global_step:%s" % (i + 1,train_acc,train_loss,train_global_step))
        
        dev_acc,dev_loss,_,_= run_epoch(session,data=Dev,model=mvalid,config=eval_config)
        print("Epoch: %d dev_acc: %.3f dev_loss %.3f" % (i + 1, dev_acc,dev_loss))

        test_acc,test_loss,_,_ = run_epoch(session, data=Test,model=mtest,config=eval_config)
        print("Epoch: %d test_acc: %.3f test_loss %.3f" % (i + 1, test_acc,test_loss))
        sys.stdout.flush()
        # if <= then update 
        if best_accuracy <= dev_acc:
          best_accuracy = dev_acc
          best_val_epoch = i
          if FLAGS.save_path:
            print("train_global_step:%s.  Saving %d model to %s." % (train_global_step,i+1,FLAGS.save_path))
            sv.saver.save(session,"model_saved/model", global_step=train_global_step)
            print (time.strftime("%Y-%m-%d %H:%M:%S",time.localtime()))

        if (i - best_val_epoch > config.update_learning)and(i-last_change_epoch>config.change_epoch):
          if learning_rate>config.min_lr:
            lr_decay = config.lr_decay ** max(i + 1 - config.max_epoch, 0.0)
            new_learning_rate = config.learning_rate * lr_decay
            last_change_epoch= i
            print("learning_rate-->change!Dang!Dang!Dang!-->%.10f"%(new_learning_rate))
            m.assign_lr(session,new_learning_rate)

        if best_test_acc <test_acc:
            best_test_acc= test_acc 
            best_test_epoch= i

        
        end_time=time.time()
        print("################# all_training time: %s one_epoch time: %s ############### " % ((end_time-t0)//60, (end_time-start_time)//60))
        if i - best_val_epoch > config.early_stopping:
          print ("best_val_epoch:%d  best_val_accuracy:%s"%(best_val_epoch+1,best_accuracy))
          print ("best_test_epoch:%d  best_test_accuracy:%s"%(best_test_epoch+1,best_test_acc))
          logging.info("Normal Early stop")
          print (time.strftime("%Y-%m-%d %H:%M:%S",time.localtime()))
          break        
        elif i == config.MAXITER-1:
       
          print ("best_val_epoch:%d  best_val_accuracy:%s"%(best_val_epoch+1,best_accuracy))
          print ("best_test_epoch:%d  best_test_accuracy:%s"%(best_test_epoch+1,best_test_acc))
          logging.info("Finishe Training")

      
if __name__ == "__main__":
  tf.app.run()
