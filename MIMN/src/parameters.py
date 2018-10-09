import argparse

parser = argparse.ArgumentParser()

datasets = ['snli', "mnli","mpe","scitail","sick"]

# Valid genres to train on. 

pa = parser.add_argument

pa("dataset_name", choices=datasets, type=str, help="Differnent datasets have different format input")
pa("--rebuiltGloveEmb",type=bool,default= True)
#pa("--datapath", type=str, default="../data/")
pa("--datapath", type=str, default="./data/")
pa("--log_path", type=str, default="./logs/")
#pa("--datapath_local", type=str, default="../data")
pa("--glove_path", type=str, default= "./data/embeddings/glove.840B.300d.txt")
pa("--embedding_dir", type=str, default="./data/embeddings/")
pa("--save_path",type=str,help="save_path",default="model_saved")
pa("--restore_model",type=str,help="manually specify a model to restore ")
pa("--matching_order",type=str,help="csm",default="csm")

pa("--sick_train_file", type=str, default="sick/sick_train.txt")
pa("--sick-test_file", type=str, default="sick/sick_test.txt")
pa("--sick-dev_file",type=str,default="sick/sick_dev.txt")
pa("--sick_glove_file",type=str,default="sick_embedding.pkl.gz")


pa("--snli_train_file", type=str, default="snli_1.0/train.txt")
pa("--snli-test_file", type=str, default="snli_1.0/test.txt")
pa("--snli-dev_file",type=str,default="snli_1.0/dev.txt")
pa("--snli_glove_file",type=str,default="snli_embedding.pkl.gz")

pa("--mpe_train_file", type=str, default="mpe/pre_mpe_train.txt")
pa("--mpe-test_file", type=str, default="mpe/pre_mpe_test.txt")
pa("--mpe-dev_file",type=str,default="mpe/pre_mpe_dev.txt")
pa("--mpe_glove_file",type=str,default="mpe_embedding.pkl.gz")

pa("--scitail_train_file", type=str, default="SciTailV1/tsv_format/scitail_1.0_train.tsv")
pa("--scitail-test_file", type=str, default="SciTailV1/tsv_format/scitail_1.0_test.tsv")
pa("--scitail-dev_file",type=str,default="SciTailV1/tsv_format/scitail_1.0_dev.tsv")
pa("--scitail_glove_file",type=str,default="scitail_embedding.pkl.gz")

pa("--mnli_train_file", type=str, default="multinli_0.9/multinli_0.9_train.txt")
pa("--mnli_dev_matched_file", type=str, default="multinli_0.9/multinli_0.9_dev_matched.txt")
pa("--mnli-dev_mismatched_file",type=str,default="multinli_0.9/multinli_0.9_dev_mismatched.txt")
pa("--mnli_glove_file",type=str,default="mnli_embedding.pkl.gz")


pa("--train_file", type=str, help='bathpath+train_filepath')
pa("--test_file", type=str, help='bathpath+test_filepath')
pa("--dev_file", type=str, help='bathpath+dev_filepath')
pa("--glove_dir", type=str,default="./data/glove/sick_glove.npy", help='glove_file')
pa("--vocab_dir",type=str,default="./data/glove/sick_vocab.txt",help='vocab_file')

pa("--init_scale",type=float,default=0.1,help="init parames scale")
pa("--learning_rate", type=float, default=0.0004, help="Learning rate for model")
pa("--min_lr", type=float, default=0.000005, help="the min Learning rate")
pa("--lr_decay", default=0.8, type=float, help='learning rate decay')
pa("--max_epoch",type=int, default=8,  help='update learning rate after max_epoch')
pa("--max_max_epoch",type=int, default=4,  help='max_max_epoch')

pa("--diff_var", type=float, default=0.0, help="factor for frobenius norm loss") ####

pa("--max_grad_norm", type=float, default=5, help='max gradient norm')
pa("--num_layers", type=int, default=1, help='number of network')
pa("--xmaxlen", type=int, default=71, help="Max premise sequence length")
pa("--ymaxlen", type=int, default=8, help="Max hypothesis sequence length")
pa("--num_classes", type=int, default=3, help="batch size") ####
pa("--hidden_units", type=int, default=300, help="hidden units number")
pa("--word_embedding_size", type=int, default=300, help="size of word embedding")
pa("--MAXITER", type=int, default=70, help="Max iterative epoches")
pa("--keep_prob", type=float, default=0.8, help='keep probability')

pa("--batch_size", type=int, default=32, help="batch size") ####
pa("--vocab_size", type=int, default=40000, help=" vocab size") ##
pa("--l2_strength", type=float, default=0.0003, help='l2 regularization ratio') ##9e-5
pa("--early_stopping",type=int, default=10,  help='update learning rate after max_epoch')
pa("--best_accuracy",type=float, default=0,  help='the best accuracy')
pa("--best_val_epoch",type=int, default=0,  help='which epoch has the best accuracy')

pa("--change_epoch",type=int, default=5,  help='change learning rate after change_epoch')
pa("--update_learning",type=int, default=5,  help='update learning rate interval')
pa("--use_gated",type=bool,default=False,help='weather use gated')
pa("--test", action='store_true', help="Call if you want to only test on the best checkpoint.")
pa("--preprocess_data_only", action='store_true', help='preprocess_data_only')
pa("--num_process_prepro", type=int, default=24, help='num process prepro')
pa("--seed",type=int, default=1234,  help='seed for random initialization')


args = parser.parse_args()
#load_parameters()
def load_parameters():
    if args.dataset_name=="sick":
        args.train_file=args.datapath+args.sick_train_file
        args.test_file=args.datapath+args.sick_test_file
        args.dev_file=args.datapath+args.sick_dev_file
        args.glove_dir=args.embedding_dir + args.sick_glove_file

    if args.dataset_name=="snli":
        args.train_file=args.datapath+args.snli_train_file
        args.test_file=args.datapath+args.snli_test_file
        args.dev_file=args.datapath+args.snli_dev_file
        args.glove_dir=args.embedding_dir + args.snli_glove_file
        args.xmaxlen= 32
        args.ymaxlen= 32

    if args.dataset_name=="mpe":
        args.train_file=args.datapath+args.mpe_train_file
        args.test_file=args.datapath+args.mpe_test_file 
        args.dev_file=args.datapath+args.mpe_dev_file
        args.glove_dir=args.embedding_dir + args.mpe_glove_file

        args.xmaxlen= 79
        args.ymaxlen= 20 #17
       # args.glove_dir=args.datapath+args.glove_dir
    if args.dataset_name=="scitail":
        args.train_file=args.datapath+args.scitail_train_file
        args.test_file=args.datapath+args.scitail_test_file
        args.dev_file=args.datapath+args.scitail_dev_file
        args.glove_dir=args.embedding_dir + args.scitail_glove_file
        args.xmaxlen= 17
        args.ymaxlen= 17 #17
        args.num_classes=2

    if args.dataset_name=="mnli":
        args.train_file=args.datapath+args.mnli_train_file
        args.test_file=args.datapath+args.mnli_dev_matched_file
        args.dev_file=args.datapath+args.mnli_dev_mismatched_file
        args.glove_dir=args.embedding_dir + args.mnli_glove_file

    return args

load_parameters()

