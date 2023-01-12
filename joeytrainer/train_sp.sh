#bash scripts/train_sentencepiece.sh en yo /netscratch/alabi/data/Lafand/joeytrainer/data/enyo/JWLf/train  ../data/enyo/spmodel 10000 10k

#where en = source language, yo = target language
#the training data to use for training  = /netscratch/alabi/data/Lafand/joeytrainer/data/enyo/JWLf/train
#../data/enyo/spmodel = directory to save the sentencepiece model
#10000 = the vocabulary size
#10k = vocabulary size literal

#train sentencepiece model of 10000 vocaulary over en-yo
bash scripts/train_sentencepiece.sh en yo /netscratch/alabi/data/Lafand/joeytrainer/data/enyo/JWLf/train ./data/enyo/spmodel 10000 10k
#train sentence piece of 20000 vocaulary over en-yo
bash scripts/train_sentencepiece.sh en yo /netscratch/alabi/data/Lafand/joeytrainer/data/enyo/JWLf/train ./data/enyo/spmodel 20000 20k
#train sentence piece of 40000 vocaulary over en-yo
bash scripts/train_sentencepiece.sh en yo /netscratch/alabi/data/Lafand/joeytrainer/data/enyo/JWLf/train ./data/enyo/spmodel 40000 40k
