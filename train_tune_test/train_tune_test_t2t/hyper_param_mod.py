import sys
import yaml
import argparse

parser = argparse.ArgumentParser(description="this script sets hyperparameters for Transformer")
parser.add_argument('--config_file', help='configuration file used by registers', required=True)
parser.add_argument('--s', help="source language", required=True)
parser.add_argument('--t', help="target language", required=True)
parser.add_argument('--st', help="st", required=True)
parser.add_argument('--train_src', help="train_src", required=True)
parser.add_argument('--dev_src', help="dev_src", required=True)
parser.add_argument('--train_tgt', help="train_tgt", required=True)
parser.add_argument('--dev_tgt', help="dev_tgt", required=True)
parser.add_argument('--glove_mat', help='glove_mat', required=True)
parser.add_argument('--bilingual_lexicon', help='bilingual_lexicon', required=True)
parser.add_argument('--synonym_api', help='synonyms powered by thesaurus', required=True)
parser.add_argument('--synonym_api2', help='synonyms powered by nltk wordnet', required=True)
parser.add_argument('--use_subword_tokenizer', help="Whether to use subword tokenizer", required=True)
parser.add_argument('--share_vocab', help="Whether to share vocab", required=True)
parser.add_argument('--vocab_size', help="Vocab size for both src and tgt", required=True)
parser.add_argument('--model_params', help="Concatenation of model parameters", required=True)
parser.add_argument('--num_heads')
args = parser.parse_args()

config = {}
for key, value in vars(args).items():
    config[key] = value

with open(args.config_file, 'w') as fw:
    yaml.dump(config, fw, default_flow_style=False)
