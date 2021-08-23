import pandas as pd
import unicodedata
import pandas as pd
import os
import jsonlines



def normalize_diacritics_text(text_string):
    """Convenience wrapper to abstract away unicode & NFC"""
    return unicodedata.normalize("NFC", text_string)


def create_dir(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


def read_file(input_file):
    print(input_file)
    with open(input_file, encoding='utf-8') as f:
        text_lines = f.read().splitlines()

    sentences = [sent for sent in text_lines if len(sent) > 0]

    return sentences

def export_json_files(output_dir, filename, df, direction='en-yo'):

    to_be_saved = []
    src_data = df['source_lang'].values
    tgt_data = df['target_lang'].values
    src_lang, tgt_lang = direction.split('-')
    N_sent = df.shape[0]
    for s in range(N_sent):
        if tgt_lang=='en':
            text_string = {"translation": {src_lang: src_data[s], tgt_lang: tgt_data[s]}}
        else:
            text_string = {"translation": {src_lang:src_data[s], tgt_lang:tgt_data[s]}}
        to_be_saved.append(text_string)

    with jsonlines.open(output_dir+filename, 'w') as writer:
        writer.write_all(to_be_saved)




def combine_texts(data_dir):

    #en_jw = read_file(data_dir + 'jw300.en')
    #yo_jw = read_file(data_dir + 'jw300.yo')

    menyo_train = pd.read_csv(data_dir+'train.tsv', sep='\t')
    en_menyo_train = menyo_train['english'].values
    yo_menyo_train = menyo_train['yoruba'].values

    menyo_dev = pd.read_csv(data_dir + 'dev.tsv', sep='\t')
    en_menyo_dev = menyo_dev['english'].values
    yo_menyo_dev = menyo_dev['yoruba'].values

    menyo_test = pd.read_csv(data_dir + 'test.tsv', sep='\t')
    en_menyo_test = menyo_test['english'].values
    yo_menyo_test = menyo_test['yoruba'].values


    ## merge data
    # Train data
    train_data_en = list(en_menyo_train) #en_jw + list(en_menyo_train)
    train_data_yo = list(yo_menyo_train) #yo_jw + list(yo_menyo_train)

    df_train_enyo = pd.DataFrame(train_data_en, columns=['source_lang'])
    df_train_enyo['target_lang'] = train_data_yo

    df_train_yoen = pd.DataFrame(train_data_yo, columns=['source_lang'])
    df_train_yoen['target_lang'] = train_data_en

    # dev data
    df_dev_enyo = pd.DataFrame(en_menyo_dev, columns=['source_lang'])
    df_dev_enyo['target_lang'] = yo_menyo_dev

    df_dev_yoen = pd.DataFrame(yo_menyo_dev, columns=['source_lang'])
    df_dev_yoen['target_lang'] = en_menyo_dev

    # test data
    df_test_enyo = pd.DataFrame(en_menyo_test, columns=['source_lang'])
    df_test_enyo['target_lang'] = yo_menyo_test

    df_test_yoen = pd.DataFrame(yo_menyo_test, columns=['source_lang'])
    df_test_yoen['target_lang'] = en_menyo_test

    # output data
    output_dir = 'data/json_files/en_yo/'
    create_dir(output_dir)

    export_json_files(output_dir, 'train.json', df_train_enyo)
    export_json_files(output_dir, 'dev.json', df_dev_enyo)
    export_json_files(output_dir, 'test.json', df_test_enyo)

    output_dir = 'data/json_files/yo_en/'
    create_dir(output_dir)

    export_json_files(output_dir, 'train.json', df_train_yoen, direction='yo-en')
    export_json_files(output_dir, 'dev.json', df_dev_yoen, direction='yo-en')
    export_json_files(output_dir, 'test.json', df_test_yoen, direction='yo-en')


if __name__ == "__main__":
    data_dir  = 'data/tsv/eng_to_yor/'
    combine_texts(data_dir)
