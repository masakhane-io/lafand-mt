### [A Few Thousand Translations Go a Long Way! Leveraging Pre-trained Models for African News Translation](https://arxiv.org/abs/2205.02022) 

This repository contains a newly created MT dataset in the news domain known as [MAFAND-MT](https://github.com/masakhane-io/lafand-mt/tree/main/data/json_files) for 16 languages and 5 existing news MT corpus. We also provide the code for [training MT models](https://github.com/masakhane-io/lafand-mt/blob/main/run_translation.py) using pre-trained models like MT5, MBART, ByT5 and M2M-100, and a [notebook](https://github.com/masakhane-io/lafand-mt/blob/main/lafand.ipynb) that can be used on Google Colab. For you to use the code, your dataset should be in json format and you have to specify the right language code, if the language code is not supported by the pre-trained model, you can use a fake language code supported. 

The code is based on HuggingFace implementation (License: Apache 2.0).

The license of the NER dataset is in [CC-BY-4.0-NC](https://creativecommons.org/licenses/by-nc/4.0/), the monolingual data have difference licenses depending on the news website license. 

### Required dependencies
* python
  * [transformers](https://pypi.org/project/transformers/) : state-of-the-art Natural Language Processing for TensorFlow 2.0 and PyTorch.
  * [sacrebleu](https://pypi.org/project/sacrebleu/) : for BLEU, ChrF evaluation
* Other requirements are listed [here](https://github.com/huggingface/transformers/blob/main/examples/pytorch/translation/requirements.txt)

```bash
pip install transformers accelerate datasets sentencepiece protobuf sacrebleu py7zr torch
```

### The MAFAND dataset includes the following languages:
- Amharic (amh)
- Ghomala (bbj)
- Ewe (ewe)
- Fon (fon)
- Hausa (hau) --- we only created lines 1 - 2767 in the train set. Others are from [Khamenei](https://www.statmt.org/wmt21/translation-task.html)
- Kinyarwanda (kin) --- only dev and test sets
- Luganda (lug)
- Luo (luo) --- currently not available due to copyright issues. 
- Mossi (mos) 
- Nigerian-Pidgin (pcm)
- Chichewa (nya)  --- only dev and test sets
- Shona (sna) --- only dev and test sets
- Swahili --- dev/test set was created, train set was obtained from [Global Voices on OPUS](https://opus.nlpl.eu/GlobalVoices.php)
- Setswana (tsn)
- Twi (twi)
- Wolof (wol)
- Xhosa (xho) --- only dev and test sets

### Existing corpus
If you use existing corpus, please cite their papers
- Igbo (ibo) --- We make use of [Igbo News MT corpus](https://github.com/IgnatiusEzeani/IGBONLP/tree/master/ig_en_mt), we defined new train/dev/test
- Yoruba (yor) --- from the news domain part of [MENYO-20k](https://github.com/uds-lsv/menyo-20k_MT), we defined new modified train/dev/test split
- Zulu (zul) --- obtained from [Umsuka on Zenodo](https://zenodo.org/record/5035171#.YvpeXHUzY5k). We use the test split, but defined a new dev split


### Adaptation of mT5 and ByT5

### Adaptation of mBART50

### Fine-tune mT5, M2M100, and byT5 using [HuggingFace Transformers](https://github.com/huggingface/transformers/tree/master/examples/pytorch/translation)

Step 0: Preprocess your text to be in json lines

Step 1: Install the necessary modules in requirments.txt

Step 2: Fine-tune the pre-trained model, you may use the *base* model e.g "google/byt5-base" and "google/mt5-base"

```
python run_translation.py \
    --model_name_or_path google/byt5-base \
    --do_train \
    --do_eval \
    --source_lang en \
    --target_lang yo \
    --source_prefix "translate English to Yoruba: " \
    --train_file data/en_yo/train.json \
    --validation_file data/en_yo/dev.json \
    --test_file data/en_yo/test.json \
    --output_dir byt5_en_yo \
    --max_source_length 200 \
    --max_target_length 200 \
    --per_device_train_batch_size=10 \
    --per_device_eval_batch_size=10 \
    --overwrite_output_dir \
    --predict_with_generate \
    --save_steps 50000 \
    --num_beams 10 \
    --do_predict
```

Testing the model:

```
python run_translation.py \
    --model_name_or_path byt5_en_yo \
    --source_lang en \
    --target_lang yo \
    --source_prefix "translate English to Yoruba: " \
    --train_file data/en_yo/train.json \
    --validation_file data/en_yo/dev.json \
    --test_file data/en_yo/test.json \
    --output_dir byt5_en_yo \
    --max_source_length 200 \
    --max_target_length 200 \
    --per_device_train_batch_size=10 \
    --per_device_eval_batch_size=10 \
    --overwrite_output_dir \
    --predict_with_generate \
    --save_steps 50000 \
    --num_beams 10 \
    --do_predict
```

For MBART and M2M100 e.g "facebook/m2m100_418M", you need to specify the language to generate using "forced_bos_token"

```
python run_translation.py \
    --model_name_or_path facebook/m2m100_418M  \
    --do_train \
    --do_eval \
    --train_file data/en_yo/train.json \
    --validation_file data/en_yo/dev.json \
    --test_file data/en_yo/test.json \
    --source_lang en \
    --target_lang yo \
    --output_dir m2m100_en_yo \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --overwrite_output_dir \
    --predict_with_generate \
    --forced_bos_token yo \
    --save_steps 50000 \
    --num_beams 10 \
    --do_predict
```
