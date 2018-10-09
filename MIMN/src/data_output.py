import numpy as np

def label_input(true_label,pred_label,input_file,output_file):
  ''' 
  '''
  if len(true_label) != len(pred_label):
    raise ValueError('len(true_label) != len(pred_label)')
  else: 
    data_num =len(true_label)
    input_txt=[]
    for l in open(input_file):
      line = l.strip().split('\t') 
      if line[2] !="-" and line[2]!="gold_label":
        #input_txt.append(line[:-1])
        input_txt.append(line)

    if len(input_txt) != data_num:
      print ("input_num",len(input_txt))
      print ("pred_num",len(pred_label))
      raise ValueError('len(input_txt) != len(pred_label)')

    fout = open(output_file,"w+")
    label_dict ={"neutral":0,"entailment":1,"contradiction":2}

    for i in range (data_num):
      if true_label[i] != label_dict[input_txt[i][2]]:
        print ("error_hint!!! true_label :%s input_txt_label:%s"%(true_label[i],input_txt[i][2]))
        raise ValueError('true_label[i] != input_txt[i][2]')

      np.set_printoptions(precision=3)
      out_line = "true:"+ str(true_label[i]) + "\tpred:"+ str(pred_label[i]) +"\t"+  str(input_txt[i][0])+"\t"+str(input_txt[i][1])+ "\t"+  str(input_txt[i][2])  +"\n"

      out_line_clean = ''.join(map(str, out_line))
      fout.write(out_line_clean)
