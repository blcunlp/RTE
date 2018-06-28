echo training
echo 1
nohup python3 ./snli_train_lr.py --save_path model_saved_1 --learning_rate 0.0004 --keep_prob 0.8 --l2_strength 0.0002 --knowledge_rate 5 --train_path '../data/snli_1.0/train_4.txt' 2>f2_1 1>f1_1
echo 2
nohup python3 ./snli_train_lr.py --save_path model_saved_2 --learning_rate 0.0004 --keep_prob 0.8 --l2_strength 0.0002 --knowledge_rate 8 --train_path '../data/snli_1.0/train_4.txt' 2>f2_2 1>f1_2
echo 3
nohup python3 ./snli_train_lr.py --save_path model_saved_3 --learning_rate 0.0004 --keep_prob 0.8 --l2_strength 0.0002 --knowledge_rate 10 --train_path '../data/snli_1.0/train_4.txt' 2>f2_3 1>f1_3
