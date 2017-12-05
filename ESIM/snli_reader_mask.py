# coding: utf-8
import tensorflow as tf
import os
import json
from myutils import *
#from keras.preprocessing.sequence import pad_sequences
#from keras.utils.np_utils import to_categorical
from collections import Counter

from six.moves import xrange
import numpy as np
_PAD="_PAD"
_UNK= "_UNK"
_GO= "_GO"
_EOS= "_EOS"
_START_VOCAB=[_PAD,_UNK,_GO,_EOS]

PAD_ID=0
UNK_ID=1
GO_ID =2
EOS_ID =3

def filter_length(seq,maxlen):
  if len(seq)>maxlen:
    new_seq=seq[:maxlen]
  else:
    new_seq=seq
  return new_seq

def load_data(train,vocab,labels={'neutral':0,'entailment':1,'contradiction':2}):
    X,Y,Z=[],[],[]
    #f_l=open("seq.txt","w+")
    for p,h,l in train:
        p=map_to_idx(tokenize(p),vocab)+ [EOS_ID]
        h=[GO_ID]+map_to_idx(tokenize(h),vocab)+ [EOS_ID]
        p=filter_length(p,32)
        h=filter_length(h,30)
        if l in labels:         # get rid of '-'
            X+=[p]
            Y+=[h]
            Z+=[labels[l]]
    return X,Y,Z

def get_vocab(data):
    vocab=Counter()
    for ex in data:
        tokens=tokenize(ex[0])
        tokens+=tokenize(ex[1])
        vocab.update(tokens)
    #lst = ["unk", "delimiter"] + [ x for x, y in vocab.iteritems() if y > 0]
    vocab_sorted = sorted(vocab.items(), key=lambda x: (-x[1], x[0]))
    lst = _START_VOCAB + [ x for x, y in vocab_sorted if y > 0]

    vocab_exist=os.path.isfile("../data/glove/vocab.txt")

    if not vocab_exist:
      print ("write vocab.txt")
      f =open("./glove/vocab.txt","w+")
      for x,y in enumerate(lst):
        x_y = str(y) +"\t"+ str(x)+"\n"
        f.write(x_y)
        #f.write("\n")
      f.close()
    
    vocab = dict([ (y,x) for x,y in enumerate(lst)])
    return vocab


class DataSet(object):
  def __init__(self,x,y,labels,x_len,y_len,X_mask,Y_mask):
    self._data_len=len(x)
    self._x =x
    self._y =y
    self._labels =labels
    self._x_len = x_len
    self._y_len = y_len
    self._epochs_completed = 0
    self._index_in_epoch = 0
    self._num_examples = x.shape[0]
    self._x_mask=X_mask
    self._y_mask=Y_mask

  def next_batch(self, batch_size):
    """Return the next `batch_size` examples from this data set."""

    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1

      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples

    end = self._index_in_epoch

    batch_x, batch_x_mask, batch_x_len = self._x[start:end], self._x_mask[start:end], self._x_len[start:end]
    batch_y,batch_y_mask, batch_y_len = self._y[start:end], self._y_mask[start:end], self._y_len[start:end]
    batch_labels = self._labels[start:end]
    
    return batch_x,batch_y, batch_labels,batch_x_mask,batch_y_mask,batch_x_len,batch_y_len

  @property
  def get_x(self):
    return self._x
  
  @property
  def get_y(self):
    return self.y

  @property
  def labels(self):
    return self._labels

  @property
  def get_x_len(self):
    return self._x_len
  
  @property
  def get_y_len(self):
    return self._y_len

  @property
  def get_data_num(self):
    return self._data_len
  
  def get_epoch_size(self,batch_size):
    epoch_size = self._data_len //batch_size
    return epoch_size

def file2seqid(config):
  
  xmaxlen = config.xmaxlen
  ymaxlen = config.ymaxlen
  train = [l.strip().split('\t') for l in open(config.train_file)]
  dev = [l.strip().split('\t') for l in open(config.dev_file)]
  test = [l.strip().split('\t') for l in open(config.test_file)]
  vocab = get_vocab(train)

  X_train, Y_train, Z_train = load_data(train, vocab)
  X_dev, Y_dev, Z_dev = load_data(dev, vocab)
  X_test, Y_test, Z_test = load_data(test,vocab)

  #X_train_lengths = [len(x) for x in X_train]
  X_train_lengths=np.asarray([len(x) for x in X_train]).reshape(len(X_train))
  X_dev_lengths = np.asarray([len(x) for x in X_dev]).reshape(len(X_dev))
  X_test_lengths = np.asarray([len(x) for x in X_test]).reshape(len(X_test))
  X_train_mask = np.asarray([np.ones(x) for x in X_train_lengths]).reshape(len(X_train_lengths))
  X_dev_mask= np.asarray([np.ones(x) for x in X_dev_lengths] ).reshape(len(X_dev_lengths))
  X_test_mask=np.asarray([np.ones(x) for x in X_test_lengths] ).reshape(len(X_test_lengths))

  Y_train_lengths = np.asarray([len(x) for x in Y_train]).reshape(len(Y_train))
  Y_dev_lengths = np.asarray([len(x) for x in Y_dev]).reshape(len(Y_dev))
  Y_test_lengths = np.asarray([len(x) for x in Y_test]).reshape(len(Y_test))
  Y_train_mask = np.asarray([np.ones(x) for x in Y_train_lengths]).reshape(len(Y_train_lengths))
  Y_dev_mask= np.asarray([np.ones(x) for x in Y_dev_lengths]).reshape(len(Y_dev_lengths))
  Y_test_mask=np.asarray([np.ones(x) for x in Y_test_lengths] ).reshape(len(Y_test_lengths))

  Z_train = to_categorical(Z_train, num_classes=config.num_classes)
  Z_dev = to_categorical(Z_dev, num_classes=config.num_classes)
  Z_test = to_categorical(Z_test, num_classes=config.num_classes)
  
  X_train = pad_sequences(X_train, maxlen=xmaxlen, value=vocab[_PAD], padding='post') ## NO NEED TO GO TO NUMPY , CAN GIVE LIST OF PADDED LIST
  X_dev = pad_sequences(X_dev, maxlen=xmaxlen, value=vocab[_PAD], padding='post')
  X_test = pad_sequences(X_test, maxlen=xmaxlen, value=vocab[_PAD], padding='post')
  Y_train = pad_sequences(Y_train, maxlen=ymaxlen, value=vocab[_PAD], padding='post')
  Y_dev = pad_sequences(Y_dev, maxlen=ymaxlen, value=vocab[_PAD], padding='post')
  Y_test = pad_sequences(Y_test, maxlen=ymaxlen, value=vocab[_PAD], padding='post')

  X_train_mask=pad_sequences(X_train_mask, maxlen=xmaxlen, value=vocab[_PAD], padding='post')
  X_dev_mask = pad_sequences(X_dev_mask , maxlen=xmaxlen, value=vocab[_PAD], padding='post')
  X_test_mask = pad_sequences(X_test_mask , maxlen=xmaxlen, value=vocab[_PAD], padding='post')
  Y_train_mask = pad_sequences(Y_train_mask, maxlen=ymaxlen, value=vocab[_PAD], padding='post')
  Y_dev_mask = pad_sequences(Y_dev_mask, maxlen=ymaxlen, value=vocab[_PAD], padding='post')
  Y_test_mask = pad_sequences(Y_test_mask, maxlen=ymaxlen, value=vocab[_PAD], padding='post')
  #print (len(X_train[1]),"padding X_train[1]", X_train[1])
  #print (len(Y_train[1]),"padding Y_train[1]", Y_train[1])
  #print (len(Z_train[1]),"padding Z_train[1]", Z_train[1])
  Train = DataSet(X_train,Y_train,Z_train,X_train_lengths,Y_train_lengths,X_train_mask,Y_train_mask)
  Dev = DataSet(X_dev,Y_dev,Z_dev,X_dev_lengths,Y_dev_lengths,X_dev_mask,Y_dev_mask)
  Test = DataSet(X_test,Y_test,Z_test,X_test_lengths,Y_test_lengths,X_test_mask,Y_test_mask) 
  #data_sets.train = DataSet(X_train,Y_train,Z_train,X_train_lengths,Y_train_lengths)
  #data_sets.dev = DataSet(X_dev,Y_dev,Z_dev,X_dev_lengths,Y_dev_lengths)
  #data_sets.test = DataSet(X_test,Y_test,Z_test,X_test_lengths,Y_test_lengths) 
  
  #train=(X_train,Y_train,Z_train,X_train_lengths,Y_train_lengths)
  #dev=(X_dev,Y_dev,Z_dev,X_dev_lengths,Y_dev_lengths)
  #test=(X_test,Y_test,Z_test,X_test_lengths,Y_test_lengths)
  #print ("X_train_lengths[1]",X_train_lengths[1])
  #print ("X_dev_lengths[1]",X_dev_lengths[1])
  #print ("X_test_lengths[1]",X_test_lengths[1])
  return Train,Dev,Test,vocab
  
  
def snli_producer(data,config,name=None):
  batch_size=config.batch_size
  xmaxlen=config.xmaxlen
  ymaxlen=config.ymaxlen
  num_classes=config.num_classes
  with tf.name_scope(name, "SNLIProducer"):
    data_len=data._data_len
    epoch_size =data_len//batch_size
    assertion = tf.assert_positive(
        epoch_size,
        message="epoch_size == 0, decrease batch_size or num_steps")
    with tf.control_dependencies([assertion]):
      epoch_size = tf.identity(epoch_size, name="epoch_size")
    i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()

    data_x = tf.convert_to_tensor(data._x)
    data_y = tf.convert_to_tensor(data._y)
    data_labels =  tf.convert_to_tensor(data._labels)
    data_x_len =  tf.convert_to_tensor(data._x_len)
    data_y_len =  tf.convert_to_tensor(data._y_len)
    data_x_mask=tf.convert_to_tensor(data._x_mask)
    data_y_mask=tf.convert_to_tensor(data._y_mask)

    x = tf.strided_slice(data_x, [i*batch_size,0], [(i+1)*batch_size,xmaxlen])
    y = tf.strided_slice(data_y, [i*batch_size,0], [(i+1)*batch_size,ymaxlen])

    x_mask=tf.strided_slice(data_x_mask, [i*batch_size,0], [(i+1)*batch_size,xmaxlen])
    y_mask=tf.strided_slice(data_y_mask, [i*batch_size,0], [(i+1)*batch_size,ymaxlen])

    label = tf.strided_slice(data_labels, [i*batch_size,0], [(i+1)*batch_size,num_classes])
    x_len = tf.strided_slice(data_x_len, [i*batch_size], [(i+1)*batch_size])
    y_len = tf.strided_slice(data_y_len, [i*batch_size], [(i+1)*batch_size])
    
    x.set_shape([batch_size,xmaxlen])
    y.set_shape([batch_size,ymaxlen])
    x_mask.set_shape([batch_size,xmaxlen])
    y_mask.set_shape([batch_size,ymaxlen])
    label.set_shape([batch_size,num_classes])
    x_len.set_shape([batch_size])
    y_len.set_shape([batch_size])
 
    x_mask=tf.cast(x_mask,tf.float32)
    y_mask=tf.cast(y_mask,tf.float32)

    return x,y,label,x_len,y_len,data_len,x_mask,y_mask

 
if __name__=="__main__":

    train=[l.strip().split('\t') for l in open('train.txt')][:20000]
    dev=[l.strip().split('\t') for l in open('dev.txt')]
    test=[l.strip().split('\t') for l in open('test.txt')]
    labels={'neutral':0,'entailment':1,'contradiction':2}

    vocab=get_vocab(train)
    #X_train,Y_train,Z_train=load_data(train,vocab)
    X_dev,Y_dev,Z_dev=load_data(dev,vocab)
    #print (len(X_train),X_train[0])
    print (len(X_dev),X_dev[0])
    print (len(Y_dev),Y_dev[0])
    print (len(Z_dev),Z_dev[0])
