import argparse

def main(args):
    
    # This creates the config file for our JoeyNMT system. It might seem overwhelming so we've provided a couple of useful parameters you'll need to update
    # (You can of course play with all the parameters if you'd like!)
    source_language = args.source_lang
    target_language = args.target_lang
    name = args.task
    #joeynmt/bpeddata
    data_path = args.data_path
    if args.finetune:
        checkpoint = args.ckpt_path.strip()
        if checkpoint == '':
            print('Checkpoint directory cannot be empty')
            return 
    else:
        checkpoint = None

    name = '%s_%s_%s' % (name, source_language, target_language)
    model_path = args.model_path 
    # Create the config
    config = """
    
    name: "{name}_transformer"
    
    data:
        src: "{source_language}"
        trg: "{target_language}"
        train: "{data_path}/train"
        dev:   "{data_path}/dev"
        test:  "{data_path}/test"
        level: "bpe"#bpe
        lowercase: False
        max_sent_length: 100
        src_vocab: "{data_path}/vocab.txt"
        trg_vocab: "{data_path}/vocab.txt"
    testing:
        beam_size: 5
        alpha: 1.0
        bpe_type: "sentencepiece"
        sacrebleu:                      # sacrebleu options
            remove_whitespace: True     # `remove_whitespace` option in sacrebleu.corpus_chrf() function (defalut: True)
            tokenize: "none"            # `tokenize` option in sacrebleu.corpus_bleu() function (options include: "none" (use for already tokenized test data), "13a" (default minimal tokenizer), "intl" which mostly does punctuation and unicode, etc) 
    
    training:
        #load_model: "{checkpoint}" # if uncommented, load a pre-trained model from this checkpoint
        random_seed: 42
        optimizer: "adam"
        normalization: "tokens"
        adam_betas: [0.9, 0.999] 
        scheduling: "plateau"           # TODO: try switching from plateau to Noam scheduling
        patience: 8                     # For plateau: decrease learning rate by decrease_factor if validation score has not improved for this many validation rounds
        learning_rate_factor: 0.5       # factor for Noam scheduler (used with Transformer)
        learning_rate_warmup: 1000      # warmup steps for Noam scheduler (used with Transformer)
        decrease_factor: 0.7
        loss: "crossentropy"
        learning_rate: 0.0002
        learning_rate_min: 0.00000001
        weight_decay: 0.0
        label_smoothing: 0.1
        batch_size: 4096
        batch_type: "token"
        eval_batch_size: 3600
        eval_batch_type: "token"
        batch_multiplier: 1
        early_stopping_metric: "ppl"
        epochs: 30                     # TODO: Decrease for when playing around and checking of working. Around 30 is sufficient to check if its working at all
        validation_freq: 1000          # TODO: Set to at least once per epoch.
        logging_freq: 100
        eval_metric: "bleu"
        model_dir: "{model_path}/{name}_transformer"
        overwrite: True #False               # TODO: Set to True if you want to overwrite possibly existing models. 
        shuffle: True
        use_cuda: True
        max_output_length: 100
        print_valid_sents: [0, 1, 2, 3]
        keep_last_ckpts: 3
    model:
        initializer: "xavier"
        bias_initializer: "zeros"
        init_gain: 1.0
        embed_initializer: "xavier"
        embed_init_gain: 1.0
        tied_embeddings: True
        tied_softmax: True
        encoder:
            type: "transformer"
            num_layers: 6
            num_heads: 8             # TODO: Increase to 4 for smaller data.
            embeddings:
                embedding_dim: 512   # TODO: Increase to 256 for smaller data.
                scale: True
                dropout: 0.
            # typically ff_size = 4 x hidden_size
            hidden_size: 512         # TODO: Increase to 256 for smaller data.
            ff_size: 2048            # TODO: Increase to 1024 for smaller data.
            dropout: 0.1
        decoder:
            type: "transformer"
            num_layers: 6
            num_heads: 8              # TODO: Increase to 4 for smaller data.
            embeddings:
                embedding_dim: 512    # TODO: Increase to 256 for smaller data.
                scale: True
                dropout: 0.
            # typically ff_size = 4 x hidden_size
            hidden_size: 512        # TODO: Increase to 256 for smaller data.
            ff_size: 2048            # TODO: Increase to 1024 for smaller data.
            dropout: 0.1
    """.format(name=name, model_path=model_path, checkpoint=checkpoint, data_path=data_path, source_language=source_language, target_language=target_language)
    with open("joeynmt/configs/transformer_{name}.yaml".format(name=name),'w') as f:
        f.write(config)


if __name__ == '__main__':
    # Create the parser
    parser = argparse.ArgumentParser(description='Argument for the JoeyNMT.')
    parser.add_argument("--source_lang", type=str, default='en', help="The source language.")
    parser.add_argument("--target_lang", type=str, default='fr', help="The target language.")
    parser.add_argument("--task", type=str, default='lafand', help="The destination directory.")
    parser.add_argument("--data_path", type=str, default='', help="The train/dev/test data directory.")
    parser.add_argument("--model_path", type=str, default='', help="The model directory.")
    parser.add_argument('--finetune', default=False, action='store_true')
    parser.add_argument("--ckpt_path", type=str, default='', help="The checkpoint directory.")
    # Execute the parse_args() method
    args = parser.parse_args()
    main(args)

