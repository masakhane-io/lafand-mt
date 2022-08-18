### [A Few Thousand Translations Go a Long Way! Leveraging Pre-trained Models for African News Translation](https://arxiv.org/abs/2205.02022) 

This repository contains a newly created MT dataset in the news domain known as [MAFAND-MT](https://github.com/masakhane-io/lafand-mt/tree/main/data/json_files) for 16 languages and 5 existing news MT corpus. We also provide the code for [training MT models](https://github.com/masakhane-io/lafand-mt/blob/main/run_translation.py) using pre-trained models like MT5, MBART, ByT5 and M2M-100, and a [notebook](https://github.com/masakhane-io/lafand-mt/blob/main/lafand.ipynb) that can be used on Google Colab. For you to use the code, your dataset should be in json format and you have to specify the right language code, if the language code is not supported by the pre-trained model, you can use a fake language code supported. 

The code is based on HuggingFace implementation (License: Apache 2.0).

The license of the MT dataset is in [CC-BY-4.0-NC](https://creativecommons.org/licenses/by-nc/4.0/), the French/English monolingual data we translated from have difference licenses depending on the news website license. 

### Required dependencies
* python
  * [transformers](https://pypi.org/project/transformers/) : state-of-the-art Natural Language Processing for TensorFlow 2.0 and PyTorch.
  * [sacrebleu](https://pypi.org/project/sacrebleu/) : for BLEU, ChrF evaluation
* Other requirements are listed [here](https://github.com/huggingface/transformers/blob/main/examples/pytorch/translation/requirements.txt)

```bash
pip install transformers accelerate datasets sentencepiece protobuf sacrebleu py7zr torch
```

### The MAFAND dataset

The dataset includes the following languages:
- Amharic (amh)
- Bambara (bam)
- Ghomala (bbj)
- Ewe (ewe)
- Fon (fon)
- Hausa (hau):  we only created lines 1 - 2767 in the train set. Others are from [Khamenei](https://www.statmt.org/wmt21/translation-task.html)
- Kinyarwanda (kin):  only dev and test sets
- Luganda (lug)
- Luo (luo): currently not available due to copyright issues. 
- Mossi (mos) 
- Nigerian-Pidgin (pcm)
- Chichewa (nya):  only dev and test sets
- Shona (sna):  only dev and test sets
- Swahili (swa): dev/test set was created, train set was obtained from [Global Voices on OPUS](https://opus.nlpl.eu/GlobalVoices.php)
- Setswana (tsn)
- Twi (twi)
- Wolof (wol)
- Xhosa (xho) --- only dev and test sets

### Existing corpus

If you use existing corpus, please cite their papers
- Igbo (ibo): We make use of [Igbo News MT corpus](https://github.com/IgnatiusEzeani/IGBONLP/tree/master/ig_en_mt), we defined new train/dev/test
- Yoruba (yor): from the news domain part of [MENYO-20k](https://github.com/uds-lsv/menyo-20k_MT), we defined new modified train/dev/test split
- Zulu (zul): obtained from [Umsuka on Zenodo](https://zenodo.org/record/5035171#.YvpeXHUzY5k). We use the test split, but defined a new dev split


### Adaptation of mT5 and ByT5
For the adaptation of mT5 and ByT5 models, we trained on 17 African languages and 3 high-resource languages (English, French and Arabic). The dataset is available on Zenodo i.e [AfroMAFT Corpus](https://zenodo.org/record/6990611#.Yv6le3UzY5k). Thank you [Jesujoba Alabi](https://ajesujoba.github.io/) for preparing the data. 

The pre-training code is available in [mt5_byt5_pre-training](https://github.com/masakhane-io/lafand-mt/tree/main/mt5_byt5_pre_training) directory. A big thank you to [Xiaoyu Shen](https://scholar.google.de/citations?user=BWfPrE4AAAAJ&hl=zh-TW) for providing the code and instructions. The pre-trained [AfriMT5](https://huggingface.co/masakhane/afri-mt5-base) and [AfriByT5](https://huggingface.co/masakhane/afri-byt5-base) are available on HuggingFace Model Hub

### Adaptation of mBART50
Similar to mT5, we performed continued pre-trained on [AfroMAFT Corpus](https://zenodo.org/record/6990611#.Yv6le3UzY5k). The instructions are provided in [mbart_pre-training](https://github.com/masakhane-io/lafand-mt/tree/main/mbart_pre-training). A big thank you to [Machel Reid](https://machelreid.github.io/) and [Ernie Chang](https://scholar.google.com/citations?user=FbR5cAMAAAAJ&hl=en). The [AfriMBART](https://huggingface.co/masakhane/afri-mbart50) model is also available on HuggingFace Model Hub

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


### BibTeX entry and citation info
```
@inproceedings{adelani-etal-2022-thousand,
    title = "A Few Thousand Translations Go a Long Way! Leveraging Pre-trained Models for {A}frican News Translation",
    author = "Adelani, David  and
      Alabi, Jesujoba  and
      Fan, Angela  and
      Kreutzer, Julia  and
      Shen, Xiaoyu  and
      Reid, Machel  and
      Ruiter, Dana  and
      Klakow, Dietrich  and
      Nabende, Peter  and
      Chang, Ernie  and
      Gwadabe, Tajuddeen  and
      Sackey, Freshia  and
      Dossou, Bonaventure F. P.  and
      Emezue, Chris  and
      Leong, Colin  and
      Beukman, Michael  and
      Muhammad, Shamsuddeen  and
      Jarso, Guyo  and
      Yousuf, Oreen  and
      Niyongabo Rubungo, Andre  and
      Hacheme, Gilles  and
      Wairagala, Eric Peter  and
      Nasir, Muhammad Umair  and
      Ajibade, Benjamin  and
      Ajayi, Tunde  and
      Gitau, Yvonne  and
      Abbott, Jade  and
      Ahmed, Mohamed  and
      Ochieng, Millicent  and
      Aremu, Anuoluwapo  and
      Ogayo, Perez  and
      Mukiibi, Jonathan  and
      Ouoba Kabore, Fatoumata  and
      Kalipe, Godson  and
      Mbaye, Derguene  and
      Tapo, Allahsera Auguste  and
      Memdjokam Koagne, Victoire  and
      Munkoh-Buabeng, Edwin  and
      Wagner, Valencia  and
      Abdulmumin, Idris  and
      Awokoya, Ayodele  and
      Buzaaba, Happy  and
      Sibanda, Blessing  and
      Bukula, Andiswa  and
      Manthalu, Sam",
    booktitle = "Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
    month = jul,
    year = "2022",
    address = "Seattle, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.naacl-main.223",
    doi = "10.18653/v1/2022.naacl-main.223",
    pages = "3053--3070",
    abstract = "Recent advances in the pre-training for language models leverage large-scale datasets to create multilingual models. However, low-resource languages are mostly left out in these datasets. This is primarily because many widely spoken languages that are not well represented on the web and therefore excluded from the large-scale crawls for datasets. Furthermore, downstream users of these models are restricted to the selection of languages originally chosen for pre-training. This work investigates how to optimally leverage existing pre-trained models to create low-resource translation systems for 16 African languages. We focus on two questions: 1) How can pre-trained models be used for languages not included in the initial pretraining? and 2) How can the resulting translation models effectively transfer to new domains? To answer these questions, we create a novel African news corpus covering 16 languages, of which eight languages are not part of any existing evaluation dataset. We demonstrate that the most effective strategy for transferring both additional languages and additional domains is to leverage small quantities of high-quality translation data to fine-tune large pre-trained models.",
}
```
