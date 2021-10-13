src=en
tgt=yo
#mkdir -p joeynmt/data/frmos/fr_mos/10k/

python joeynmt/scripts/build_vocab.py /netscratch/alabi/data/Lafand/joeytrainer/data/enyo/spdata/10k/JW/train.$src /netscratch/alabi/data/Lafand/joeytrainer/data/enyo/spdata/10k/JW/train.$tgt --output_path /netscratch/alabi/data/Lafand/joeytrainer/data/enyo/spdata/10k/JW/vocab.txt

python joeynmt/scripts/build_vocab.py /netscratch/alabi/data/Lafand/joeytrainer/data/enyo/spdata/20k/JW/train.$src /netscratch/alabi/data/Lafand/joeytrainer/data/enyo/spdata/20k/JW/train.$tgt --output_path /netscratch/alabi/data/Lafand/joeytrainer/data/enyo/spdata/20k/JW/vocab.txt
