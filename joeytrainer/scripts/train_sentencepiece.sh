# To install Sentencepiece with Conda
#conda install -c conda-forge sentencepiece 

#To train the sentence piece model training and encoding
src=$1 #e.g. en the source language code
tgt=$2 #.e.g. yo the target language code

datadir=$3 #e.g. data/JWLf/train , the directory containing the training data 
combinedir="${datadir}.all"  #e.g. data/JWLf/train.all
spdir=$4 #e.g. data/spmodel, the directory to store the sentencepiece model
vocab_size=$5 #e.g.10000,  the vocabulary size to use for the sentencepiece model
vocab_name=$6 #e.g. 10k, the vocabulary size name, .e.g 10k for 10000, 1k for 1000 etc

#Combine the train dataset into one file to learn joint sentence piece model
cat $datadir.$src $datadir.$tgt > $combinedir

mkdir -p $spdir
#Train the model on Combined english-arabic dataset 
spm_train --input=$combinedir --model_prefix="$spdir/sp$src$tgt$vocab_name" --vocab_size=$vocab_size --character_coverage=1.0  --max_sentence_length=4096

for lang in $src $tgt; do
	#get the sentencepiece vocabulary from the training data. 
	spm_encode --model="$spdir/sp$src$tgt$vocab_name.model" --generate_vocabulary < $datadir.$lang > $spdir/vocab.$vocab_name.$lang
done



