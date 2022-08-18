from transformers import AutoTokenizer
import random
from tqdm import tqdm
import sys

def racha_detection(lista):
    # It returns a list of lists where each sub-list contains the consecutive tokens in the list
    rachas = []
    racha = []
    for i, element in enumerate(lista):
        if (i<len(lista)-1) and (lista[i+1] == element+1):
            racha.append(element)
        else:
            if len(racha)>0:
                rachas.append(racha + [element])          
            else:# (i!=len(lista)-1):
                rachas.append([element])
            racha = []
    return rachas

def masking(tokenized_sentence, rachas):
    # Function to mask a tokenized_sentence (token ids) following the rachas described in rachas
    # Only one sentinel_token per racha
    sent_token_id = 0
    enmascared = tokenized_sentence.copy()
    for racha in rachas:
        sent_token = f'<extra_id_{sent_token_id}>'
        sent_id = tokenizer.encode(sent_token)[0]
        for i, idx in enumerate(racha):
            if i==0:
                enmascared[idx] = sent_id
            else:
                enmascared[idx] = -100
        sent_token_id += 1
    
    enmascared = [t for t in enmascared if t!=-100] 

    return enmascared

def add_noise(sentence, tokenizer, percent=0.15):
    # Function that takes a sentence, tokenizer and a noise percentage and returns
    # the masked input_ids and masked target_ids accordling with the T5 paper and HuggingFace docs
    # To see the process working uncomment all the prints ;)
    tokenized_sentence = tokenizer.encode(sentence)
    #print('PRE-MASKED:')
    #print('INPUT: {}'.format(tokenizer.convert_ids_to_tokens(tokenized_sentence)))

    idxs_2_mask = sorted(random.sample(range(len(tokenized_sentence)), 
                                       int(len(tokenized_sentence)*percent)))
    rachas = racha_detection(idxs_2_mask)
    enmascared_input = masking(tokenized_sentence, rachas)
    #print('RACHAS INPUT: {}'.format(rachas))
    idxs_2_mask = [idx for idx in range(len(tokenized_sentence)) if idx not in idxs_2_mask]
    rachas = racha_detection(idxs_2_mask)
    enmascared_target = masking(tokenized_sentence, rachas)
    #print('RACHAS TARGET: {}'.format(rachas))

    #print('POST-MASKED:')
    #print('INPUT: {}'.format(tokenizer.convert_ids_to_tokens(enmascared_input)))
    #print('TARGET: {}'.format(tokenizer.convert_ids_to_tokens(enmascared_target)))

    return enmascared_input, enmascared_target

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("google/byt5-base")
    files = ['af',  'am',  'ar',  'en',  'fr',  'ha',  'ig',  'mg',  'ny',  'om',  'pcm',  'rw',  'sn',  'so',  'st',  'sw',  'xh',  'yo',  'zu']
    sources, targets = [], []
    for f in files:
        print(f)
        lines = open('Processed/' + f + '/train.'+f).readlines()
        for line in tqdm(lines):
            line = line.strip()
            source, target = add_noise(line, tokenizer)
            while(target[0]!=258):
                source, target = add_noise(line, tokenizer)
            sources.append(' '.join(list(map(str, source))).strip() + '\n')
            targets.append(' '.join(list(map(str, target))).strip() + '\n')
    with open('train.source' + '.' + str(i), 'w') as f:
        f.writelines(sources)
    with open('train.target' + '.' + str(i), 'w') as f:
        f.writelines(targets)
    #lines = open('Processed/af/eval.af').readlines()
    #for i in range(1000):
    #    s, t = add_noise(lines[i], tokenizer)
    #    sources.append(' '.join(list(map(str, s))).strip() + '\n')
    #    targets.append(' '.join(list(map(str, t))).strip() + '\n')
    #with open('train.source', 'w') as f:
    #    f.writelines(sources)
    #with open('train.target', 'w') as f:
    #    f.writelines(targets)
