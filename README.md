## LAFAND-MT: Lacuna Anglo &amp; Franco Africa News Dataset for low-resourced MT

### Fine-tune mT5, M2M100, and byT5 using [HuggingFace Transformers](https://github.com/huggingface/transformers/tree/master/examples/pytorch/translation)

Step 0: Preprocess your text to be in json lines (see preprocess_mt5.py), you will find an example for Yoruba-English.

Step 1: Install the necessary modules in requirments.txt

Step 2: Fine-tune the model, you may use the *base* model e.g "google/byt5-base", "google/mt5-base" and "facebook/m2m100_418M"

CUDA_VISIBLE_DEVICES=0 python run_translation.py \
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
