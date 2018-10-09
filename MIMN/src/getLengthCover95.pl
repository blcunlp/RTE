#注：1需要将代码本身单独设置成utf8的格式
use Encode;#2
use utf8; #3出现在代码中的常量按照字符流来处理，一旦有常量就一定要use utf8
open(in,'../../data/mpe/pre_mpe_train.txt');
print "had been in ... to find the length which can cover %95 training data";

	while(<in>){ 
	chomp;
#        print "$_\n";
	#$UTF8=decode("utf8",$_);
	#if($UTF8=~/([^\t]+)\t([^\t]++)\t([^\t]++)\t([^\t]+)(.*)(TRAIN)$/){#匹配训练句子的长度
	#if(/([^\t]+)\t([^\t]++)\t([^\t]++)\t(.*)/){
	if(/([^\t]+)\t([^\t]+)\t(.*)/){	
	#	print"$1	$2	$3";
		#$len=length($2);
		#$p=$2;#$2表示p
		$p=$1;#$3表示h
		$h=$2;
		$p_len=0;$h_len=0;
		@p_list=split(/\W+/,$p);foreach(@p_list){
		#print "$_ ";
		$p_len++;}
		@h_list=split(/\W+/,$h);foreach(@h_list){
		$h_len++;} 
		#$len=length(@p_list);
		#这一句显示所有句子的长度
	#	print"$p_len	$h_len\n";
		$Myp_length{$p_len}++;
		$Myh_length{$h_len}++;
		$p_all++;
		$h_all++;
		#统计总单词数
		#$all_length=$all_length+$len;
		#print "$all:$2 ";
		
	}
}
print "\nthere are $p_all P  and $h_all H scentences\n";
#print "the sum of all sentences' length $all_length\n";
close(in);
@p_keys = sort { $b <=> $a } keys %Myp_length;#for (@keys){print  "$_ -> $Mylength{$_}\n"}
@h_keys=sort {$b<=> $a} keys %Myh_length;
$i=1;$j=1;
while(1){
	$p_sum=$p_sum+$Myp_length{$i};
	#print"PPPPP:$i $Myp_length{$i}\n";
	#print"$all*0.95";
	$h_sum=$h_sum+$Myh_length{$j};
	if(($p_sum>($p_all*0.95))&&($f!=10)){
	$P_95=$i;$f=10;
	print "the P'sentences whose length is and above $i take parts %95 of all P data\n";}
	if(($h_sum>($h_all*0.95))&&($f2!=11)){
	$H_95=$j;$f2=11;
	print "the H'sentences whose length is and above $j take parts %95 of all H data\n";}
	$i++;$j++;
	if(($f==10)&&($f2==11)){last;}
}
#输出不同长度句子的数量,只写个sort是按照字母顺序排序
foreach (sort {$a <=> $b }keys %Myp_length){
	print"$_ $Myp_length{$_}\t";
	print"$_ $Myh_length{$_}\n";
}

