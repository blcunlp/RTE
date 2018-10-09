
'''
function:preprocess data.I need to extract gold_label,sentence1,sentence2

in_data_format:
gold_label	sentence1_binary_parse	sentence2_binary_parse	sentence1_parse	sentence2_parse	sentence1	sentence2	captionID	pairID	label1	label2	label3	label4	label5

out_data_format: sentence1	sentence2	gold_label
A person on a horse jumps over a broken down airplane.	A person is training his horse for a competition.	neutral

'''

import numpy as np
import random
import re
import fileinput


def snli_sen_label(infile,outfile,data_num=None):
  line_count=0
  def split_image_id(raw_sen):
    tmp = raw_sen.split("/")   # split("#") is not a good choice
    if len(tmp)==2:
      sen = tmp[1]
      return sen
    else: 
      print ("bad data-","line:",line_count,raw_sen)

  f_out =file(outfile,"w")
  #f_label=file("label.txt","w+")
  count=0
  out_list=[]
  if data_num == None:
    for line in fileinput.input(infile):
      line_count= line_count +1
      if line_count>1 and line:
        line_list=line.split("	")
        if len(line_list) ==10:
          line_out=str(split_image_id(line_list[1]))+ "	" +str(split_image_id(line_list[2]))+"	" + str(split_image_id(line_list[3]))+"	"+ str(split_image_id(line_list[4]))+"	"+str(line_list[5])+"	"+ str(line_list[-1])
          out_list.append(line_out)
          #label=str(line_list[0])+"\n"
          #f_label.write(label)
        #else:
        #  print (line)

    for line in out_list:
      count =count+1
      #print (count)
      f_out.write(line)
    print (count)


if __name__ == "__main__":
  snli_sen_label("mpe_train.txt","train.txt")
  snli_sen_label("mpe_dev.txt","dev.txt")
  snli_sen_label("mpe_test.txt","test.txt")
