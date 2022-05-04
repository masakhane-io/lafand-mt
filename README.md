## MAFAND-MT: A Few Thousand Translations Go a Long Way! Leveraging Pre-trained Models for African News Translation 

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
