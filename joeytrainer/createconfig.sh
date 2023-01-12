#create config file to train a model over the JW300 en-yo corpus using 10k sp vocabulary 
python scripts/createconfig.py --source_lang en --target_lang yo --task 10kjw300enyo --model_path  /netscratch/alabi/data/Lafand/joeytrainer/joeynmt/ --data_path /netscratch/alabi/data/Lafand/joeytrainer/data/enyo/spdata/10k/JW/
#create config file to train a model over the JW300 en-yo corpus using 10k sp vocabulary
python scripts/createconfig.py --source_lang yo --target_lang en --task 10kjw300yoen --model_path  /netscratch/alabi/data/Lafand/joeytrainer/joeynmt/ --data_path /netscratch/alabi/data/Lafand/joeytrainer/data/enyo/spdata/10k/JW/

python scripts/createconfig.py --source_lang yo --target_lang en --task 10kjw300yoenFTN --model_path  /netscratch/alabi/data/Lafand/joeytrainer/joeynmt/ --data_path /netscratch/alabi/data/Lafand/joeytrainer/data/enyo/spdata/10k/JW/ --finetune
