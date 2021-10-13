# To encode the dataset using already trained Sentencepiece model

src=$1 #e.g. en, the source language code
tgt=$2 #e.g. yo, the target language code

datadir=$3 #e.g.data/enyo
spdir=$datadir/spmodel/ #e.g. data/enyo/spmodel/, the directory containing the already trained sentencepiece model
spdatadir=$datadir/spdata/ #.e.g. data/enyo/spdata/ #the diretory to save the encoded sentencepiece model
vocab_size=$4
vocab_name=$5 #the name of the sentencepiece model as used in the sentencepiece.sh file

mkdir -p $spdatadir

for item in JW JWLf Lafand; do
	data="${datadir}/${item}"
	wdir=${spdatadir}/${vocab_name}/${item}
	mkdir -p $wdir
	for  dataset in train dev test; do
		filesrc=${data}/${dataset}.$src
		filetgt=${data}/${dataset}.$tgt	
		if [ -e $filesrc -a -e $filetgt ]; then
			echo "both exist"
			#spm_encode --model="./$spdir/sp$src$tgt$vocab_name.model" --generate_vocabulary < $datadir.$lang > $spdir/vocab.$vocab_name.$lang	
			spm_encode --model="$spdir/sp$src$tgt$vocab_name.model" --vocabulary=$spdir/vocab.$vocab_name.$src < $filesrc > ${wdir}/${dataset}.${src}
			spm_encode --model="$spdir/sp$src$tgt$vocab_name.model" --vocabulary=$spdir/vocab.$vocab_name.$tgt < $filetgt > ${wdir}/${dataset}.${tgt}
			#apply sentencepiece on the data 
		else
			echo "${filesrc}  or ${filetgt} is missing "
		fi
	done
done

