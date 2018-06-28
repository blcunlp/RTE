import tensorflow as tf
import tensorflow.contrib.cudnn_rnn as cudnn_rnn
from itertools import zip_longest
import queue
import threading
import numpy as np

def m_cudnn_lstm(inputs, num_layers, hidden_size, is_training, weight_noise_std=None):
    """Run the CuDNN LSTM.
    Arguments:
        - inputs:   A tensor of shape [batch, max_sents_num,max_sent_length, input_size] of inputs.
        - num_layers:   Number of RNN layers.
        - hidden_size:     Number of units in each layer.
        - is_training:     tf.bool indicating whether training mode is enabled.
    Return a tuple of (outputs, init_state, final_state).
    """
    
    flat_inputs = flatten(inputs, 2)  # [-1, J, d]
    flat_outputs,output_h,output_c = cudnn_lstm(flat_inputs, num_layers, hidden_size, weight_noise_std, is_training)
    outputs = reconstruct(flat_outputs, inputs, 2)
    return outputs, output_h, output_c


def cudnn_lstm(inputs, num_layers, hidden_size, is_training, weight_noise_std=None,scope=None):
    """Run the CuDNN LSTM.
    Arguments:
        - inputs:   A tensor of shape [batch, length, input_size] of inputs.
        - layers:   Number of RNN layers.
        - hidden_size:     Number of units in each layer.
        - is_training:     tf.bool indicating whether training mode is enabled.
    Return a tuple of (outputs, init_state, final_state).
    """
    input_size = inputs.get_shape()[-1].value
    if input_size is None:
        raise ValueError("Number of input dimensions to CuDNN RNNs must be "
                         "known, but was None.")

    # CUDNN expects the inputs to be time major
    inputs = tf.transpose(inputs, [1, 0, 2])

    cudnn_cell = tf.contrib.cudnn_rnn.CudnnLSTM(
        num_layers, hidden_size, input_size,
        input_mode="linear_input", direction="bidirectional")

    est_size = estimate_cudnn_parameter_size(
        num_layers=num_layers,
        hidden_size=hidden_size,
        input_size=input_size,
        input_mode="linear_input",
        direction="bidirectional")

    cudnn_params = tf.get_variable(
        "RNNParams",
        shape=[est_size],
        initializer=tf.contrib.layers.variance_scaling_initializer())

    if weight_noise_std is not None:
        cudnn_params = weight_noise(
            cudnn_params,
            stddev=weight_noise_std,
            is_training=is_training)
    # initial_state: a tuple of tensor(s) of shape`[num_layers * num_dirs, batch_size, num_units]
    init_state = tf.tile(
        tf.zeros([2 * num_layers, 1, hidden_size], dtype=tf.float32),
        [1, tf.shape(inputs)[1], 1])  # [2 * num_layers, batch_size, hidden_size]
    '''
    Args:
      inputs: `3-D` tensor with shape `[time_len, batch_size, input_size]`.
      initial_state: a tuple of tensor(s) of shape
        `[num_layers * num_dirs, batch_size, num_units]`. If not provided, use
        zero initial states. The tuple size is 2 for LSTM and 1 for other RNNs.
      training: whether this operation will be used in training or inference.
    Returns:
      output: a tensor of shape `[time_len, batch_size, num_dirs * num_units]`.
        It is a `concat([fwd_output, bak_output], axis=2)`.
      output_states: a tuple of tensor(s) of the same shape and structure as
        `initial_state`.
    '''
    hiddens, output_h, output_c = cudnn_cell(
        inputs,
        input_h=init_state,
        input_c=init_state,
        params=cudnn_params,
        is_training=True)

    # Convert to batch major
    hiddens = tf.transpose(hiddens, [1, 0, 2])
    output_h = tf.transpose(output_h, [1, 0, 2])
    output_c = tf.transpose(output_c, [1, 0, 2])

    return hiddens, output_h, output_c


def cudnn_lstm_parameter_size(input_size, hidden_size):
    """Number of parameters in a single CuDNN LSTM cell."""
    biases = 8 * hidden_size
    weights = 4 * (hidden_size * input_size) + 4 * (hidden_size * hidden_size)
    return biases + weights


def direction_to_num_directions(direction):
    if direction == "unidirectional":
        return 1
    elif direction == "bidirectional":
        return 2
    else:
        raise ValueError("Unknown direction: %r." % (direction,))


def estimate_cudnn_parameter_size(num_layers,
                                  input_size,
                                  hidden_size,
                                  input_mode,
                                  direction):
    """
    Compute the number of parameters needed to
    construct a stack of LSTMs. Assumes the hidden states
    of bidirectional LSTMs are concatenated before being
    sent to the next layer up.
    """
    num_directions = direction_to_num_directions(direction)
    params = 0
    isize = input_size
    for layer in range(num_layers):
        for direction in range(num_directions):
            params += cudnn_lstm_parameter_size(
                isize, hidden_size
            )
        isize = hidden_size * num_directions
    return params

def parameter_count():
    """Return the total number of parameters in all Tensorflow-defined
    variables, using `tf.trainable_variables()` to get the list of
    variables."""
    return sum(np.product(var.get_shape().as_list())
               for var in tf.trainable_variables())

def flatten(tensor, keep):
    fixed_shape = tensor.get_shape().as_list()
    start = len(fixed_shape) - keep
    left = reduce(mul, [fixed_shape[i] or tf.shape(tensor)[i] for i in range(start)])
    out_shape = [left] + [fixed_shape[i] or tf.shape(tensor)[i] for i in range(start, len(fixed_shape))]
    flat = tf.reshape(tensor, out_shape)
    return flat

def reconstruct(tensor, ref, keep):
    ref_shape = ref.get_shape().as_list()  # context:[N,M,JX,d] question:[N,JQ,d]
    tensor_shape = tensor.get_shape().as_list()  # context:[N*M,JX,d] question:[N,JQ,d]
    ref_stop = len(ref_shape) - keep  # context:2,question:1
    tensor_start = len(tensor_shape) - keep  # 1 
    pre_shape = [ref_shape[i] or tf.shape(ref)[i] for i in range(ref_stop)]
    keep_shape = [tensor_shape[i] or tf.shape(tensor)[i] for i in range(tensor_start, len(tensor_shape))]
    # pre_shape = [tf.shape(ref)[i] for i in range(len(ref.get_shape().as_list()[:-keep]))]
    # keep_shape = tensor.get_shape().as_list()[-keep:]
    target_shape = pre_shape + keep_shape
    out = tf.reshape(tensor, target_shape)
    return out
