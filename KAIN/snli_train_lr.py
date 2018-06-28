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
import copy
import tensorflow as tf
from tensorflow.contrib.rnn.python.ops import core_rnn_cell
from tensorflow.contrib.layers import batch_norm,l2_regularizer
from tensorflow.python.ops import variable_scope
from sklearn.metrics import confusion_matrix
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
flags.DEFINE_string("save_path","model_saved",
                    "Model output directory.")
flags.DEFINE_bool("use_fp16", False,
                  "Train using 16-bit floats instead of 32bit floats")
flags.DEFINE_float('learning_rate',0.0003,'initial learning rate')
flags.DEFINE_float('keep_prob',0.8,'keep_prob for dropout')
flags.DEFINE_float('l2_strength',0.0003,'l2 rate for l2 loss')
flags.DEFINE_float('knowledge_rate',5,'the rate add to the attention weight')
flags.DEFINE_string('train_path',"../data/snli_1.0/train.txt",'the train data path')
flags.DEFINE_bool('use_feature',False,'wether use feature')


FLAGS = flags.FLAGS

def data_type():
  return tf.float16 if FLAGS.use_fp16 else tf.float32

def fill_placeholder(data, model,config):
  batch_x,batch_y,batch_label,batch_x_mask,batch_y_mask, batch_x_len,batch_y_len,feature_x,mask_feature_x_inside,mask_feature_x_outside= data.next_batch(config.batch_size)
  feature_x=np.reshape(feature_x,(config.batch_size,config.xmaxlen,config.ymaxlen,4)) 
  #feature=np.reshape(feature,(32,32,30,5)) 
  feed_dict = {model.x:batch_x , 
                model.y:batch_y,
                model.label:batch_label,
                model.x_mask:batch_x_mask,
                model.y_mask:batch_y_mask, 
                model.x_len :batch_x_len,
                model.y_len :batch_y_len,
                model.feature_x:feature_x,
                model.mask_feature_x_inside:mask_feature_x_inside,
                model.mask_feature_x_outside:mask_feature_x_outside,
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
  pred_prod_total=[]
  v1_all=[]
  v2_all = []
  cos_dis = []
  fetches = {
      "acc":model.acc,
      "loss": model.loss,
      "global_step":model.global_step,
      "learning_rate": model.learning_rate,
      "v":model.v,
      "pred_label":model.pred,
      "true_label":model.label,
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
    pred_label_total.extend(np.argmax(vals["pred_label"],1))
    true_label_total.extend(np.argmax(vals["true_label"],1))
    pred_prod_total.extend(vals["pred_label"])
    cos_dis.extend(cosine_distance(v1=np.split(vals["v"][0],2)[0], v2 =np.split(vals["v"][0],2)[1] ))
    #if verbose and step %10 == 0:
    #  print('global_step: %s train_acc: %s  batch_train_loss: %s' % (global_step,acc,loss))
    acc_average=acc_total/iters
    loss_average = losses/iters
    #print(feature[0][0][0])

  confuse_m = confusion_matrix(y_true = true_label_total , y_pred =pred_label_total )
  confuse_m_normalize = confuse_m.astype('float') / confuse_m.sum(axis=1)[:, np.newaxis] 

  #print ("confusion_matrix\n",confuse_m)
  np.set_printoptions(precision=3)
  #print ("confusion_matrix_normalize\n",confuse_m_normalize)
  return acc_average,loss_average,global_step,learning_rate,confuse_m,confuse_m_normalize

def cosine_distance(v1,v2):
  ''' 
  v1:[batch_size,dim]
  v2:[batch_size,dim]
  '''
  v1 = v1.astype('float32')
  v2 = v2.astype('float32')
  #v1.get_shape().assert_is_compatible_with(v2.get_shape())

  v1_norm = np.sqrt(np.sum(np.square(v1), axis=-1,keepdims=True)) #(b,1)
  v2_norm = np.sqrt(np.sum(np.square(v2), axis=-1,keepdims=True)) #(b,1)

  radial_diffs = v1*v2 #(b,dim)
  numerator = np.sum(radial_diffs, axis=-1, keepdims=True) #(b,1)
  cosine = numerator /(v1_norm * v2_norm+ 1e-10)  #(b,1)/(b,1)=>(b,1)

  return cosine


def get_config():
  if FLAGS.model == "small":
    return SmallConfig()
  elif FLAGS.model == "test":
    return TestConfig()
  else:
    raise ValueError("Invalid model: %s", FLAGS.model)


def main(_):
  config = get_config()
  config.learning_rate=FLAGS.learning_rate
  config.save_path=FLAGS.save_path 
  config.keep_prob=FLAGS.keep_prob
  config.l2_strength=FLAGS.l2_strength
  config.knowledge_rate=FLAGS.knowledge_rate
  config.train_file=FLAGS.train_path
  config.use_feature=FLAGS.use_feature

  eval_config=copy.deepcopy(config)
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
        train_acc,train_loss,train_global_step,learning_rate,_,_ = run_epoch(session,data=Train, model=m,config=config, eval_op=m.optim, verbose=True)
        print("Epoch: %d train_acc: %.3f train_loss %.3f train_global_step:%s learning_rate:%.3f" % (i + 1,train_acc,train_loss,train_global_step,learning_rate))
        
        dev_acc,dev_loss,_,_,_,_= run_epoch(session,data=Dev,model=mvalid,config=eval_config)
        print("Epoch: %d dev_acc: %.3f dev_loss %.3f" % (i + 1, dev_acc,dev_loss))
        test_acc,test_loss,_,_,confuse_m,confuse_m_normalize = run_epoch(session, data=Test,model=mtest,config=eval_config)
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
            print ("confusion_matrix\n",confuse_m)
            print ("confusion_matrix_normalize\n",confuse_m_normalize)

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
