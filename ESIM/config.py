class SmallConfig(object):
  """Small config."""
  init_scale = 0.1
  #learning_rate = 0.001
  #learning_rate = 0.0001
  #learning_rate = 0.0003
  learning_rate = 0.0004
  min_lr = 0.000005
  lr_decay = 0.8
  max_epoch = 8
  max_max_epoch = 4

  max_grad_norm = 5
  num_layers = 1
  xmaxlen=32
  ymaxlen=30
  num_classes=3
  hidden_units = 300
  embedding_size =300
  MAXITER=70
  #keep_prob = 1.0
  #keep_prob = 0.9
  keep_prob = 0.8
  #keep_prob = 0.7
  #keep_prob = 0.5
              
  #batch_size = 128
  batch_size = 32
  vocab_size = 40000
  #l2_strength=0.001
  #l2_strength=0.0001
  l2_strength=0.0003
  early_stopping=10
  best_accuracy = 0
  best_val_epoch = 0
 
  change_epoch = 5
  update_learning = 5
  train_file='../data/snli_1.0/train.txt'
  #train_file='../data/snli_1.0/train_20W1.txt'
  #train_file='../data/snli_1.0/train_10W.txt'
  dev_file='../data/snli_1.0/dev.txt'
  test_file='../data/snli_1.0/test.txt'
  
class TestConfig(object):
  """Tiny config, for testing."""
  init_scale = 0.1
  learning_rate = 0.001
  min_lr = 0.000005
  lr_decay = 0.8
  max_epoch = 8
  max_max_epoch = 4

  max_grad_norm = 5
  num_layers = 1
  xmaxlen=32
  ymaxlen=30
  num_classes=3
  hidden_units = 150
  embedding_size=300
  MAXITER=50
  keep_prob = 1.0

  batch_size = 10
  vocab_size=10000
  l2_strength=0.001
  #l2_strength=0.0001
  #l2_strength=0.0003

  change_epoch = 5
  update_learning = 5

  early_stopping=10
  best_accuracy = 0
  best_val_epoch = 0
  train_file='../data/snli_1.0/train_100.txt'
  dev_file='../data/snli_1.0/dev_10.txt'
  test_file='../data/snli_1.0/test_10.txt'

