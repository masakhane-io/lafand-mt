## Running Lafand-MT experiments

1. `bash setup.sh`:
If you don't have Joey NMT already installed on your system, this file creates a virtual environment called jnmt, clones the joeynmt github page and install the needed requirement. 

2. `bash ProcessData.sh` (optional):
Incase any your dataset is still in the tsv format, containing parallel texts, then it has to be separated to independent files for each of the languages. This the format that can be used for training. 

3. `bash train_sp.sh`:
The sentencepiece.sh script is responsible for installing the sentencepiece package, training a sentence piece model and obtaining the vocabulary for the sentencepiece model. 
 - uncomment the second line in the file to install sentencepiece
 - Change the variable values to suit your need
To train a sentence piece of a particular vocabulary size for a language pair, you can use the following command, check the `train_sp.sh` and change the command line argument accordingly. 

4. `bash apply_sp.sh`:
encoding the training/dev/test sets using the already the trained model. 
Change the variable values to suit your need. 

Check the `apply_sp.sh` file and make necessary changes to suit your need.

## To train the models. 
1. `bash createconfig.sh`: Create a configuration file. It contains the model parameters to use for training.

2. `bash buildvocab.sh`: Get the vocabulary needed by JoeyNMT

3. `bash train.sh`: Run the train script



 


