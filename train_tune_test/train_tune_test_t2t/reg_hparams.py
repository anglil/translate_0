from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import sys 

from tensor2tensor.models import transformer
from tensor2tensor.utils import registry
# an example of the data generator
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.data_generators import tokenizer
from tensor2tensor.data_generators import generator_utils
import tensorflow as tf
from collections import defaultdict
import yaml


@registry.register_hparams
def transformer_all():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    config_file = os.path.join(dir_path, "config.yml")
    config = yaml.load(open(config_file))
    hparams = transformer.transformer_base()

    hparams.num_heads = int(config["num_heads"])

    model_name = config["model_params"].split("_")[0]
    assert(model_name.startswith("t2t"))
    if  model_name != "t2t":
        hparams.input_modalities = "inputs:symbol:lex_modality"
        hparams.target_modality = "symbol:lex_modality"
        hparams.lex_cap = 4
        #hparams.filter_size = 1024
        hparams.num_heads = 4

    hparams.use_pad_remover = True if "withpadding" in config["model_params"] else False
    #hparams.shared_embedding_and_softmax_weights = int(False)
    hparams.glove_mat = config["glove_mat"]
    hparams.bilingual_lexicon = config["bilingual_lexicon"]
    hparams.synonym_api = config["synonym_api"]
    hparams.num_encoder_layers = int(config["model_params"].split('_')[2][5:])
    hparams.num_decoder_layers = int(config["model_params"].split('_')[2][5:])
    hparams.num_hidden_layers = int(config["model_params"].split('_')[2][5:]) # should be superceded by num_encoder_layers and num_decoder_layers
    hparams.hidden_size = int(config["model_params"].split('_')[1][3:])
    hparams.dropout = float(config["model_params"].split('_')[4][7:])
    hparams.learning_rate = float(config["model_params"].split('_')[3][2:])
    hparams.batch_size = 2048
    hparams.clip_grad_norm = 2.0
    return hparams
