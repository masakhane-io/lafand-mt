src=en #the source language code
tgt=yo #the target langauge code
name=lafand  #the name of the task as used in the createconfig.sh
cd joeynmt;
mkdir -p /netscratch/alabi/data/Lafand/joeytrainer/joeynmt/${name}_${src}_${tgt}_transformer/

srun -p batch --ntasks 1 --gpus-per-task 1 python3 -m joeynmt train configs/transformer_${name}_${src}_${tgt}.yaml 
