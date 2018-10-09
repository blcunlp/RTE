# -*- coding: utf-8 -*-
import numpy as np
import re
import random
import json
import gzip
import pickle
import collections
import numpy as np
from tqdm import tqdm
import nltk
from nltk.corpus import wordnet as wn 
import os
import pickle
import multiprocessing
import parameters


config = parameters.load_parameters()


def generate_embeddings(pretrain_embedding_path,word_indices,embedding_file_name):
  embedding_dir = config.embedding_dir
  if not os.path.exists(config.embedding_dir):
      os.makedirs(embedding_dir)
  
  #embedding_path = os.path.join(embedding_dir,embedding_file_name)
  embedding_path = embedding_file_name
  
  #logger.Log("embedding path:%s %r"%(embedding_path, os.path.exists(embedding_path)))
  print ("embedding path:%s %r"%(embedding_path, os.path.exists(embedding_path)))
  if os.path.exists(embedding_path) and config.rebuiltGloveEmb== False:
      f = gzip.open(embedding_path, 'rb')
      loaded_embeddings = pickle.load(f)
      f.close()
  else:
      loaded_embeddings = loadEmbedding_rand(pretrain_embedding_path, word_indices)
      f = gzip.open(embedding_path, 'wb')
      pickle.dump(loaded_embeddings, f)
      f.close()

  return loaded_embeddings



def loadEmbedding_rand(pretrain_embedding_path, word_indices, divident = 1.0): # TODO double embedding
    """
    Load GloVe embeddings. Doing a random normal initialization for OOV words.
    """
    j = 0
    n = len(word_indices)
    print ("vocab has", len(word_indices), "entries (including _PAD or _UNK or _GO or _EOS)")

    m = config.word_embedding_size
    emb = np.empty((n, m), dtype=np.float32)

    emb[:,:] = np.random.normal(size=(n,m)) / divident

    # Explicitly assign embedding of <PAD> to be zeros.
    emb[0, :] = np.zeros((1,m), dtype="float32")
    
    tokens_requiring_random = set(word_indices.keys())
    with open(pretrain_embedding_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            s = line.split()
            if s[0] in word_indices:
                try:
                    emb[word_indices[s[0]], :] = np.asarray(s[1:])
                    tokens_requiring_random.remove(s[0])
                except ValueError:
                    print(s[0])
                    continue

    print ("after passing over glove there are %d tokens requiring a random alloc"%( len(tokens_requiring_random)))
    return emb
