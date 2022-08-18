from datasets import load_metric
import sys
import torch.nn as nn
from tqdm import tqdm
from util import *
from argparse import ArgumentParser
import numpy as np
from transformers import QuestionAnsweringPipeline
import pandas as pd
import pickle
import shutil
import os
import time
from scipy.special import softmax

def eval_outputs(of, reference, batch_size, metric):
    metric = load_metric(metric)
    outs, refs = open(of).readlines(), open(reference).readlines()
    compare_chunk = chunks(list(zip(outs, refs)), batch_size)
    for o in tqdm(list(compare_chunk)):
        os = [m[0] for m in o]
        rs = [m[1] for m in o]
        preds = [l.strip().split() for l in os]
        tgts = [[l.strip().split()] for l in rs]
        metric.add_batch(predictions=preds, references=tgts)
    scores = metric.compute()
    print(scores)

def gen_eval(model, tokenizer, dloader, output_dir, max_len, args, testtype):
    if args.compute_metric:
        metric = load_metric(args.metric)
    outputs = []
    model.config.force_bos_token_to_be_generated = False
    with torch.no_grad():
        for inputs in tqdm(list(dloader)):
            #print('inputs:',inputs)
            #sys.exit()
            inputs = {k: v.cuda() for k, v in inputs.items()}
            if args.decode == 'beam_search':
                preds = model.generate(inputs["input_ids"], num_beams=args.num_beams, num_return_sequences=args.num_samples,max_length=max_len, early_stopping=True, do_sample = False, decoder_start_token_id = model.config.bos_token_id)
            elif args.decode == 'nucleus':
                preds = model.generate(inputs["input_ids"], do_sample=True, top_p = 0.8, num_return_sequences=args.num_samples, max_length=max_len, early_stopping=True, decoder_start_token_id = model.config.bos_token_id)
            preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip() for g in preds]
            outputs.extend([l + "\n" for l in preds])
            if 'bleu' in args.metric:
                preds = [l.split() for l in preds]
            labels = inputs['labels']
            labels[labels < 0] = tokenizer.pad_token_id
            trgs = [[tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip().split()] for g in labels] if 'bleu' in args.metric else [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip() for g in labels]
            if args.compute_metric:
                metric.add_batch(predictions=preds, references=trgs)
    if args.compute_metric:
        scores = metric.compute()
        outputs.append(str(scores) + '\n')
    with open(Path(output_dir).joinpath('decode.' + testtype), "w") as f:
        f.writelines(outputs)
