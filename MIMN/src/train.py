# ==============================================================================
# Function: train model
# Author: Chunhua Liu
# date: 20180707
# ==============================================================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time
import inspect
import logging
import copy
import json
import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn.python.ops import core_rnn_cell
from tensorflow.contrib.layers import batch_norm,l2_regularizer
from tensorflow.python.ops import variable_scope

from load_embeddings import generate_embeddings
from myutils import confuse_matrix
import data_reader as reader
import parameters 
from logger import Logger
from mimn import MyModel



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
  true_label_total=[]
  pred_label_total=[]
  fetches = {
      "acc":model.acc,
      "loss": model.loss,
      "global_step":model.global_step,
      "learning_rate": model.learning_rate,
      "pred": model.pred,
      "label": model.label,
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
    
    pred = vals["pred"]
    label = vals["label"]

    losses += loss
    iters= iters+1
    acc_total += acc
    pred_label_total.extend(np.argmax(vals["pred"],1))
    true_label_total.extend(np.argmax(vals["label"],1))
    if verbose and step %1000 == 0:
      print('global_step: %s train_acc: %s  batch_train_loss: %s' % (global_step,acc,loss))
    acc_average=acc_total/iters
    loss_average = losses/iters

  return acc_average,loss_average,global_step,learning_rate,pred_label_total,true_label_total


def main(_):
  config=parameters.load_parameters()
  eval_config= copy.deepcopy(config)
  eval_config.batch_size=1
  config_json = json.dumps(vars(config), indent=4, sort_keys=True)

  if not os.path.exists(config.log_path):
    os.makedirs(config.log_path)

  if config.test:
    logpath = "{}/{}".format(config.log_path, config.save_path) + "_test.log"
  else:
    logpath = "{}/{}".format(config.log_path, config.save_path) + ".log"
	
  logger = Logger(logpath)

  ckpt_file = os.path.join(config.log_path, config.save_path) + "/model"


  logger.Log ('config: %s'%(config_json,)) 

  Train,Dev,Test,vocab = reader.file2seqid(config)
  pretrain_embedding = generate_embeddings(config.glove_path,vocab, config.glove_dir) 
  tf.set_random_seed(config.seed)
  with tf.Graph().as_default():
    #initializer = tf.contrib.layers.xavier_initializer(uniform=True,seed=None,dtype=tf.float32)
    initializer = tf.random_uniform_initializer(-config.init_scale,config.init_scale)
    with tf.name_scope("Train"):
      with tf.variable_scope("Model", reuse=None, initializer=initializer):
        m = MyModel(is_training=True, config=config,pretrain_embedding=pretrain_embedding)
    
    with tf.name_scope("Valid"):
      with tf.variable_scope("Model", reuse=True, initializer=initializer):
        mvalid = MyModel(is_training=False,config=eval_config,pretrain_embedding=pretrain_embedding)

    with tf.name_scope("Test"):
      with tf.variable_scope("Model", reuse=True, initializer=initializer):
        mtest = MyModel(is_training=False, config=eval_config,pretrain_embedding=pretrain_embedding)
    
    #sv = tf.train.Supervisor(logdir=config.save_path)
    sv = tf.train.Supervisor()
    with sv.managed_session() as session:
      logger.Log("\n\nmodel params:%s"%(np.sum([np.product([xi.value for xi in x.get_shape()]) for x in tf.trainable_variables()])))
      t0=time.time()
 
      best_accuracy = config.best_accuracy
      best_val_epoch = config.best_val_epoch
      last_change_epoch = 0

      best_test_acc=0
      best_test_epoch=0
      
      #global train_acc
      for i in range(config.MAXITER):
        start_time=time.time()
        train_acc,train_loss,train_global_step,learning_rate,train_pred_total,train_true_total= run_epoch(session,data=Train, model=m,config=config, eval_op=m.optim, verbose=True)
        logger.Log("Epoch: %d train_acc: %.3f train_loss %.3f train_global_step:%s" % (i,train_acc,train_loss,train_global_step))

        dev_acc,dev_loss,_,_,dev_pred_total,dev_true_total= run_epoch(session,data=Dev,model=mvalid,config=eval_config)
        logger.Log("Epoch: %d dev_acc: %.3f dev_loss %.3f" % (i, dev_acc,dev_loss))

      
        sys.stdout.flush()
        # if <= then update 
        if best_accuracy <= dev_acc:
          best_accuracy = dev_acc
          best_val_epoch = i
          if config.save_path:
            logger.Log("Saving model %d to %s." % (i,ckpt_file))
            
            sv.saver.save(session,ckpt_file, global_step=train_global_step)

        if (i - best_val_epoch > config.update_learning)and(i-last_change_epoch>config.change_epoch):
          if learning_rate>config.min_lr:
            lr_decay = config.lr_decay ** max(i - config.max_epoch, 0.0)
            new_learning_rate = config.learning_rate * lr_decay
            last_change_epoch= i
            logger.Log("learning_rate-->change!Dang!Dang!Dang!-->%.10f"%(new_learning_rate))
            m.assign_lr(session,new_learning_rate)

          logger.Log (time.strftime("%Y-%m-%d %H:%M:%S",time.localtime()))

        
        end_time=time.time()
        logger.Log("-------- all_training time: %s one_epoch time: %s\n " % ((end_time-t0)//60, (end_time-start_time)//60))
        if i - best_val_epoch > config.early_stopping:
          logger.Log ("best_val_epoch:%d  best_val_accuracy:%.3f"%(best_val_epoch,best_accuracy))
          logging.info("Normal Early stop")
          logger.Log (time.strftime("%Y-%m-%d %H:%M:%S",time.localtime()))
          break        
        elif i == config.MAXITER-1:
          logger.Log ("best_val_epoch:%d  best_val_accuracy:%.3f"%(best_val_epoch,best_accuracy))
          logging.info("Finishe Training")

      logger.Log("\n\n-----------------------testing!!!-----------------------" )
      #### evaluate on the test set
      #1. restore the best parameters
      #2. test on the test set and logger.Log confusion matrix 
      #ckpt = tf.train.get_checkpoint_state(config.save_path)
      ckpt = tf.train.get_checkpoint_state(ckpt_file)
      if ckpt and ckpt.model_checkpoint_path: 
        #init_step = int(ckpt.model_checkpoint_path.rsplit('-',1)[1])
        logger.Log ("restore best model:%s for testing:"%(ckpt.model_checkpoint_path))
        sv.saver.restore(session, ckpt.model_checkpoint_path)

      test_acc,test_loss,_,_,test_pred_total,test_true_total = run_epoch(session, data=Test,model=mtest,config=eval_config)
      logger.Log ("best_test_accuracy:%.3f test_loss %.3f "%(test_acc,test_loss))
      logger.Log("\nthe confuse_matrix of test:\n")
      logger.Log(confuse_matrix(true_label_total=test_true_total,pred_label_total=test_pred_total,config=config))
      logger.Log (time.strftime("%Y-%m-%d %H:%M:%S",time.localtime()))

if __name__ == "__main__":
  tf.app.run()
