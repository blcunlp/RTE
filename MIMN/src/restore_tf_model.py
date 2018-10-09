# ==============================================================================
# Function: restore a pre-trained tensorflow model
# Author: Chunhua Liu
# date: 20180614
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
import data_reader as reader
from mimn import MyModel
import parameters 
from data_output import label_input
from myutils import confuse_matrix


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
  #global acc_average
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
    #print('global_step: %s acc: %s' % (global_step,acc))
    #print ("train and test one ITER  time",(time.time()-start_time)/60.0)
    acc_total += acc
    pred_label_total.extend(np.argmax(vals["pred"],1))
    true_label_total.extend(np.argmax(vals["label"],1))
    #if verbose and step %10 == 0:
    #  print('global_step: %s train_acc: %s  batch_train_loss: %s' % (global_step,acc,loss))
    acc_average=acc_total/iters
    loss_average = losses/iters



  return acc_average,loss_average,global_step,learning_rate,pred_label_total,true_label_total

def main(_):
  config=parameters.load_parameters()
  eval_config= copy.deepcopy(config)
  eval_config.batch_size=1
  config_json = json.dumps(vars(config), indent=4, sort_keys=True)
  #print ("config",config_json) 

  Train,Dev,Test,vocab = reader.file2seqid(config)
  pretrain_embedding = generate_embeddings(config.glove_path,vocab, config.glove_dir) 
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
      
 
    sv = tf.train.Supervisor() #logdir用来保存checkpoint和summary
    with sv.managed_session() as session:
      print ("model params",np.sum([np.product([xi.value for xi in x.get_shape()]) for x in tf.trainable_variables()]))

      model_existed=False
      if config.restore_model !=None:
        print ("restore model from: %s "%(config.restore_model))
        sv.saver.restore(session, config.restore_model)
        model_existed=True
      else:
        #ckpt = tf.train.get_checkpoint_state(config.save_path)
        ckpt_file = os.path.join(config.log_path, config.save_path) 
        ckpt = tf.train.get_checkpoint_state(ckpt_file)
        if ckpt and ckpt.model_checkpoint_path: 
          print ("restore model from: %s "%(ckpt.model_checkpoint_path))
          sv.saver.restore(session, ckpt.model_checkpoint_path)
          init_step = int(ckpt.model_checkpoint_path.rsplit('-',1)[1])
          model_existed=True
        
       #dev_acc,dev_loss,_,_,dev_pred,dev_label,dev_pred_total,dev_true_total= run_epoch(session,data=Dev,model=mvalid,config=eval_config)
       #print("Epoch: %d dev_acc: %.3f dev_loss %.3f" % (i + 1, dev_acc,dev_loss))
      if model_existed==True:
        test_acc,test_loss,_,_,test_pred_label,test_true_label = run_epoch(session, data=Test,model=mtest,config=eval_config)
        print("\n\n-----------------------testing!!!-----------------------" )
        print("test_acc: %.3f test_loss %.3f" % (test_acc,test_loss))
        print("\nthe confuse_matrix of test:\n")
        print(confuse_matrix(true_label_total=test_true_label,pred_label_total=test_pred_label,config=config))
        #label_input(true_label=test_true_label,
        #      pred_label=test_pred_label,
        #      input_file=config.test_file,
        #      output_file=FLAGS.label_out_file)   

      else:
        print ("pretrained model does not exist")
              
if __name__ == "__main__":
  tf.app.run()
