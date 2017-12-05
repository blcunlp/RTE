import sys
import re
import numpy as np
import argparse
import random
import string
# from keras import backend as K
def tokenize(sent):
    '''
    data_reader.tokenize('a#b')
    ['a', '#', 'b']
    '''
    #return [x.strip().lower() for x in re.split('(\W+)?', sent) if x.strip()]
    return [x.strip().lower() for x in re.split('(\W+)', sent) if x.strip()]


def detokenize(tokens):
    """Slightly cleaner version of joining with spaces.
    convert ['hello','world','!','hahahha'] to 'hello world! hahaha'
    Args:
        tokens (list<string>): the sentence to print
    Return:
        str: the sentence
    """
    detoken = ""
    for t in tokens:
        #print ("t:",t)
        detoken += str(t) + " "

    detoken = detoken.strip(" ")
    return detoken


def map_to_idx(x, vocab):
    '''
    x is a sequence of tokens
    '''
    # 1 is for UNK,0 is for PAD
    return [ vocab[w] if w in vocab else 1 for w in x  ]

def map_to_txt(x,vocab):
    textify=map_to_idx(x,inverse_map(vocab))
    return ' '.join(textify)
    
def inverse_map(vocab):
    return {v: k for k, v in vocab.items()}

def id_batch2txt_batch(id_batch, vocab):
    id2word=inverse_map(vocab)
    txt_batch = []
    for id_seq in id_batch:
        txt_seq=[]
        for wordId in id_seq:
            if wordId == 3:  # End of generated sentence
                break
            #elif wordId != 0 and wordId != 2:
            else:
                txt_seq.append(id2word[wordId])
        txt_seq_detoken = detokenize(txt_seq)
        txt_batch.append(txt_seq_detoken)
    return txt_batch


def inverse_ids2txt(X_inp,Y_inp,vocabx,vocaby,outp=None):
    '''
    takes x,y int seqs and maps them back to strings
    '''
    inv_map_x = inverse_map(vocabx)
    inv_map_y = inverse_map(vocaby)
    if outp:
        for x,y,z in zip(X_inp,Y_inp,outp):
            print(' '.join(map_to_idx(x,inv_map_x)))
            print(' '.join(map_to_idx(y,inv_map_y)))
            print(z)
    else:
        for x,y in zip(X_inp,Y_inp):
            print(' '.join(map_to_idx(x,inv_map_x)))
            print(' '.join(map_to_idx(y,inv_map_y)))


def to_categorical(y, num_classes=None):
    """from keras.utils.np_utils import to_categorical

    Converts a class vector (integers) to binary class matrix.
    E.g. for use with categorical_crossentropy.
    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.
    # Returns
        A binary matrix representation of the input.
    """
    y = np.array(y, dtype='int').ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes))
    categorical[np.arange(n), y] = 1

    return categorical

def pad_sequences(sequences, maxlen=None, dtype='int32',
                  padding='pre', truncating='pre', value=0.):

    """from keras.preprocessing.sequence.pad_sequences
    Pads each sequence to the same length (length of the longest sequence).

    If maxlen is provided, any sequence longer
    than maxlen is truncated to maxlen.
    Truncation happens off either the beginning (default) or
    the end of the sequence.

    Supports post-padding and pre-padding (default).

    # Arguments
        sequences: list of lists where each element is a sequence
        maxlen: int, maximum length
        dtype: type to cast the resulting sequence.
        padding: 'pre' or 'post', pad either before or after each sequence.
        truncating: 'pre' or 'post', remove values from sequences larger than
            maxlen either in the beginning or in the end of the sequence
        value: float, value to pad the sequences to the desired value.

    # Returns
        x: numpy array with dimensions (number_of_sequences, maxlen)

    # Raises
        ValueError: in case of invalid values for `truncating` or `padding`,
            or in case of invalid shape for a `sequences` entry.
    """
    if not hasattr(sequences, '__len__'):
        raise ValueError('`sequences` must be iterable.')
    lengths = []
    for x in sequences:
        if not hasattr(x, '__len__'):
            raise ValueError('`sequences` must be a list of iterables. '
                             'Found non-iterable: ' + str(x))
        lengths.append(len(x))

    num_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break

    x = (np.ones((num_samples, maxlen) + sample_shape) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if not len(s):
            continue  # empty list/array was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" not understood' % truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x



if __name__=="__main__":
    pass
